# DCP Updated — Point Cloud Registration with Deep Closest Point

Register two real-world `.ply` point clouds (e.g. Stanford Bunny / Dragon scans) using a pretrained **Deep Closest Point (DCP)** model — no ModelNet40 dataset pipeline required.

The original DCP codebase only works with synthetic ModelNet40 pairs (random rotation/translation applied to the same object). This repo adds `register_ply.py`, a standalone script that takes **two independent `.ply` scans**, feeds them through a pretrained DCP network, and outputs the rigid transform (rotation + translation) that aligns the source to the target.

## Features

- Loads and registers arbitrary `.ply` point clouds (not just ModelNet40 HDF5 data)
- Auto-normalizes point clouds to the unit sphere scale DCP was trained on, then rescales results back to original coordinates
- Computes registration quality metrics: **fitness** and **inlier RMSE** (same method as Open3D's `evaluate_registration`)
- Saves a before/after 3D visualization (`.png`), with optional inline display in Jupyter/Colab
- Optional interactive Open3D viewer window (`--show`, local display only)
- Includes sample scans: Stanford Bunny (`bun000.ply`, `bun045.ply`) and Dragon (`dragonStandRight_*.ply`, `happyStandRight_*.ply`)

## Requirements

```bash
pip install torch numpy scipy matplotlib open3d
```

You'll also need `model.py` and `util.py` from the DCP repo (defines the `DCP` network and `transform_point_cloud`), plus a pretrained checkpoint (see `pretrained/`).

## Usage

```bash
python register_ply.py \
  --ply1 bun000.ply \
  --ply2 bun045.ply \
  --model_path pretrained/dcp_v1.t7 \
  --emb_nn dgcnn \
  --pointer transformer \
  --head svd \
  --show
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--ply1` | *required* | Source point cloud (the one that gets transformed) |
| `--ply2` | *required* | Target point cloud (the one being aligned to) |
| `--model_path` | *required* | Path to pretrained DCP checkpoint (`.t7`) |
| `--num_points` | `1024` | Points sampled from each cloud for network input |
| `--emb_nn` | `dgcnn` | Embedding network: `pointnet` or `dgcnn` |
| `--pointer` | `transformer` | Attention module: `identity` or `transformer` |
| `--head` | `svd` | Registration head: `mlp` or `svd` |
| `--output` | `registered.ply` | Path to save the aligned source cloud |
| `--vis_output` | `registration_result.png` | Path to save the before/after plot |
| `--no_vis` | off | Skip generating the visualization |
| `--show` | off | Open an interactive Open3D window (requires a display) |
| `--max_corr_dist` | 5% of target extent | Distance threshold for inlier counting in metrics |

> ⚠️ `--emb_nn`, `--pointer`, `--head`, and the model dimension args (`--emb_dims`, `--n_blocks`, `--n_heads`, `--ff_dims`, `--dropout`) **must match** the configuration the checkpoint was trained with, or `load_state_dict` will fail.

## How it works

1. **Load** both `.ply` files as raw point arrays.
2. **Normalize** each cloud independently to fit a unit sphere (DCP was trained on ModelNet40 meshes at this scale — real scans are in meters/mm and need rescaling first).
3. **Subsample** each cloud to a fixed `--num_points` for the network.
4. **Run DCP inference** to predict rotation `R` and translation `t` (source → target, in normalized space).
5. **Apply the transform** to the *full-resolution* source cloud, then rescale back into the target's original coordinate frame.
6. **Evaluate** registration quality via nearest-neighbor fitness/RMSE.
7. **Save** the registered cloud and a before/after visualization.

## Output

- `registered.ply` — source cloud transformed into the target's frame
- `registration_result.png` — side-by-side before/after scatter plot
- Console output: predicted `R`/`t`, inference time, fitness, and inlier RMSE

## Notes

- Works on CPU or CUDA (`--no_cuda` to force CPU).
- `--show` only works with a local display; on headless servers/Colab it automatically falls back to the saved PNG.
- Based on [Deep Closest Point (Wang & Solomon, ICCV 2019)](https://github.com/WangYueFt/dcp).
