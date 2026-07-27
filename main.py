#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Register two .ply point clouds with a pretrained DCP model.

This does NOT go through ModelNet40 / the Dataset class in data.py at all —
that pipeline only reads ModelNet HDF5 files and synthesizes its own random
rotation/translation for training pairs. For real .ply scans you already
have two clouds and just want the transform between them, so this script
loads them directly and calls the DCP net the same way main.py does at
inference time (net(src, target) -> rotation_ab, translation_ab, ...).

Usage:
    pip install open3d --no-cache-dir   # if not already installed
    python register_ply.py \
        --ply1 bun000.ply --ply2 bun045.ply \
        --model_path checkpoints/dcp_v2/models/model.best.t7 \
        --emb_nn dgcnn --pointer transformer --head svd \
        --output registered.ply
"""

import argparse
import os
import time
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
from scipy.spatial import cKDTree

from model import DCP
from util import transform_point_cloud


def _in_notebook():
    """True if running inside an IPython/Jupyter/Colab kernel (e.g. invoked
    via `%run register_ply.py ...` in a Colab cell) -- as opposed to a plain
    `!python register_ply.py ...` subprocess call, where there's no kernel
    to display inline output back to."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def load_ply_points(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError(f"No points read from {path} -- check the file/format.")
    return pts


def normalize_to_unit_sphere(pts):
    """
    DCP is trained on ModelNet40 meshes that are pre-normalized to fit in a
    unit sphere. Real scanner output (e.g. the Stanford bunny, in meters/mm)
    is at a totally different scale, so the network will not behave sensibly
    on raw coordinates. We center on the centroid and scale by the max
    radius, and return the parameters needed to undo this afterward.
    """
    centroid = pts.mean(axis=0, keepdims=True)
    centered = pts - centroid
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale == 0:
        scale = 1.0
    normalized = centered / scale
    return normalized, centroid, scale


def sample_points(pts, num_points, seed=None):
    rng = np.random.RandomState(seed)
    n = pts.shape[0]
    if n >= num_points:
        idx = rng.choice(n, num_points, replace=False)
    else:
        # not enough points -- sample with replacement to hit num_points
        idx = rng.choice(n, num_points, replace=True)
    return pts[idx]


def _subsample_for_plot(pts, max_points=3000, seed=0):
    """Plotting every point in a full-res scan is slow and unreadable --
    thin it out just for the figure (doesn't touch the actual output ply)."""
    if pts.shape[0] <= max_points:
        return pts
    rng = np.random.RandomState(seed)
    idx = rng.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


def _set_equal_aspect(ax, all_pts):
    """matplotlib 3D axes don't auto-equalize aspect ratio, so a rotation
    can look skewed/wrong even when it's correct. Force equal scale."""
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = np.max(maxs - mins) / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def visualize_registration(src_before, tgt, src_after, save_path, point_size=1.5, inline=None):
    """
    Side-by-side before/after plot:
      left  = source (red) and target (blue) before alignment
      right = registered source (red) and target (blue) after alignment
    Always saved as a PNG (works headlessly). If `inline` is True (or left
    as None and a notebook kernel is detected -- e.g. Colab, Jupyter, or
    `%run register_ply.py ...`), the figure is also rendered inline in the
    notebook output instead of only living on disk.
    """
    src_before_p = _subsample_for_plot(src_before)
    src_after_p = _subsample_for_plot(src_after)
    tgt_p = _subsample_for_plot(tgt)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(*src_before_p.T, c='red', s=point_size, label='source (unaligned)')
    ax1.scatter(*tgt_p.T, c='blue', s=point_size, label='target')
    ax1.set_title('Before registration')
    _set_equal_aspect(ax1, np.concatenate([src_before_p, tgt_p], axis=0))
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(*src_after_p.T, c='red', s=point_size, label='source (registered)')
    ax2.scatter(*tgt_p.T, c='blue', s=point_size, label='target')
    ax2.set_title('After DCP registration')
    _set_equal_aspect(ax2, np.concatenate([src_after_p, tgt_p], axis=0))
    ax2.legend(loc='upper right')

    for ax in (ax1, ax2):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved before/after visualization to {save_path}")

    if inline is None:
        inline = _in_notebook()

    if inline:
        # In a notebook, showing the already-built figure re-renders it in
        # the cell output (Colab's inline backend does this automatically;
        # this call is what makes it appear when the backend isn't already
        # inline, e.g. after `%run` with no `%matplotlib inline` set).
        plt.show()
    else:
        plt.close(fig)


def compute_registration_metrics(transformed_src, tgt, max_correspondence_distance):
    """
    DCP doesn't give you known point-to-point correspondences the way ICP
    does, so 'fitness'/'inlier RMSE' here are computed the same way Open3D's
    registration.evaluate_registration does it: for each transformed source
    point, find its nearest neighbor in the target via a KD-tree.
      fitness     = fraction of source points with a target neighbor within
                    max_correspondence_distance (higher is better, max 1.0)
      inlier_rmse = RMSE of distances, over inlier pairs only (lower is
                    better; only meaningful if fitness isn't tiny)
    """
    tree = cKDTree(tgt)
    distances, _ = tree.query(transformed_src, k=1)
    inliers = distances < max_correspondence_distance
    fitness = float(np.mean(inliers))
    if inliers.sum() > 0:
        inlier_rmse = float(np.sqrt(np.mean(distances[inliers] ** 2)))
    else:
        inlier_rmse = float('nan')
    return fitness, inlier_rmse


def build_args_namespace(cli_args):
    """DCP's constructor (model.py) expects an args object with these fields --
    they must match whatever the checkpoint was trained with, or
    load_state_dict will mismatch on shapes."""
    ns = argparse.Namespace()
    ns.emb_nn = cli_args.emb_nn
    ns.pointer = cli_args.pointer
    ns.head = cli_args.head
    ns.emb_dims = cli_args.emb_dims
    ns.n_blocks = cli_args.n_blocks
    ns.n_heads = cli_args.n_heads
    ns.ff_dims = cli_args.ff_dims
    ns.dropout = cli_args.dropout
    ns.cycle = False
    return ns


def main():
    parser = argparse.ArgumentParser(description="Register two .ply files with pretrained DCP")
    parser.add_argument('--ply1', type=str, required=True, help='Path to source .ply (A)')
    parser.add_argument('--ply2', type=str, required=True, help='Path to target .ply (B)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model.t7')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'])
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'])
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'])
    parser.add_argument('--emb_dims', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--ff_dims', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=str, default='registered.ply',
                         help='Where to save src transformed into target frame')
    parser.add_argument('--vis_output', type=str, default='registration_result.png',
                         help='Where to save the before/after visualization PNG')
    parser.add_argument('--no_vis', action='store_true', default=False,
                         help='Skip generating the visualization')
    parser.add_argument('--inline', dest='inline', action='store_true', default=None,
                         help='Force displaying the plot inline (for a notebook/Colab cell)')
    parser.add_argument('--no_inline', dest='inline', action='store_false',
                         help='Force NOT displaying the plot inline, only save the PNG')
    parser.add_argument('--show', action='store_true', default=False,
                         help='Also open an interactive Open3D window '
                              '(only works with a local display, not on a headless server/Colab)')
    parser.add_argument('--max_corr_dist', type=float, default=None,
                         help='Distance threshold (in the target .ply\'s own units) for counting a '
                              'point as an inlier when computing fitness/RMSE. Default: 5%% of the '
                              'target cloud\'s extent.')
    cli_args = parser.parse_args()

    device = torch.device('cuda' if (torch.cuda.is_available() and not cli_args.no_cuda) else 'cpu')

    # --- 1. Load raw points -------------------------------------------------
    src_raw = load_ply_points(cli_args.ply1)
    tgt_raw = load_ply_points(cli_args.ply2)

    # --- 2. Normalize to the scale DCP was trained on -----------------------
    src_norm, src_centroid, src_scale = normalize_to_unit_sphere(src_raw)
    tgt_norm, tgt_centroid, tgt_scale = normalize_to_unit_sphere(tgt_raw)

    # --- 3. Subsample to num_points (network expects a fixed-size input) ----
    src_pts = sample_points(src_norm, cli_args.num_points, seed=cli_args.seed)
    tgt_pts = sample_points(tgt_norm, cli_args.num_points, seed=cli_args.seed + 1)

    # data.py feeds points into the network as (3, N), batch of 1 -> (1, 3, N)
    src_t = torch.from_numpy(src_pts.T).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(tgt_pts.T).unsqueeze(0).to(device)

    # --- 4. Build net + load pretrained weights -----------------------------
    net_args = build_args_namespace(cli_args)
    net = DCP(net_args).to(device)
    state_dict = torch.load(cli_args.model_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    # --- 5. Inference --------------------------------------------------------
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        rotation_ab, translation_ab, rotation_ba, translation_ba = net(src_t, tgt_t)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    registration_time = time.time() - start_time

    R = rotation_ab.squeeze(0).cpu().numpy()       # 3x3, src -> tgt (in normalized space)
    t = translation_ab.squeeze(0).cpu().numpy()    # 3,

    print("Predicted rotation (A -> B, normalized space):")
    print(R)
    print("Predicted translation (A -> B, normalized space):")
    print(t)
    print(f"DCP registration time: {registration_time * 1000:.2f} ms (device: {device})")

    # --- 6. Apply the transform to the FULL src cloud, then undo normalization
    src_full_norm = (src_raw - src_centroid) / src_scale
    src_full_t = torch.from_numpy(src_full_norm.T.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        transformed_norm = transform_point_cloud(
            src_full_t, rotation_ab, translation_ab
        ).squeeze(0).cpu().numpy().T  # back to (N, 3)

    # rescale back into target's original coordinate frame
    transformed_full = transformed_norm * tgt_scale + tgt_centroid

    # --- 6b. Registration quality: fitness + inlier RMSE --------------------
    max_corr_dist = cli_args.max_corr_dist
    if max_corr_dist is None:
        tgt_extent = np.max(tgt_raw.max(axis=0) - tgt_raw.min(axis=0))
        max_corr_dist = 0.05 * tgt_extent
    fitness, inlier_rmse = compute_registration_metrics(transformed_full, tgt_raw, max_corr_dist)
    print(f"Registration fitness: {fitness:.4f} "
          f"(fraction of source points within {max_corr_dist:.6g} of a target point)")
    print(f"Registration inlier RMSE: {inlier_rmse:.6g}")

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(transformed_full.astype(np.float64))
    o3d.io.write_point_cloud(cli_args.output, out_pcd)
    print(f"Saved registered source cloud to {cli_args.output}")

    # --- 7. Visualization ----------------------------------------------------
    if not cli_args.no_vis:
        visualize_registration(
            src_before=src_raw,
            tgt=tgt_raw,
            src_after=transformed_full,
            save_path=cli_args.vis_output,
            inline=cli_args.inline,
        )

    if cli_args.show:
        if os.environ.get('DISPLAY') is None and os.environ.get('XDG_RUNTIME_DIR') is None:
            print(f"[--show] No display detected (this looks like a headless server/Colab session) "
                  f"-- skipping the interactive window. The static plot at {cli_args.vis_output} "
                  f"is still your result; download it or open it there instead.")
        else:
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(transformed_full.astype(np.float64))
            src_pcd.paint_uniform_color([0, 1, 0])  # green = registered source
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(tgt_raw.astype(np.float64))
            tgt_pcd.paint_uniform_color([0.15, 0.15, 0.15])  # gray = target
            o3d.visualization.draw_geometries(
                [src_pcd, tgt_pcd], 
                window_name="DCP Registration Result",
                width=1600, 
                height=1200
            )


if __name__ == '__main__':
    main()
