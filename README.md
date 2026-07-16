# Orion JAX — Neural Star Remover (JAX/Flax)

Orion is a neural network that removes stars from deep‑sky astrophotography images, producing a "starless" version that can be used for processing nebulosity and background structures. This repository is a JAX/Flax (nnx) port of an earlier PyTorch implementation.

Highlights
- Pure JAX/Flax (nnx) — no TensorFlow in the data pipeline
- U‑Net encoder–decoder built from NAFNet blocks (LayerNorm + SimpleGate, no BatchNorm)
- Hybrid ViT‑conv bottleneck with learned positional embeddings and stochastic depth
- PixelShuffle upsampling; SE (Squeeze‑Excitation) skip gates
- Multi‑component loss (Charbonnier, log‑Charbonnier, SSIM, gradient, frequency)
- Per‑run output directory with checkpointing, persisted metrics history, and a side‑by‑side run comparison script


## Quick start

1) Install Python and dependencies
- Python: 3.13+
- Recommended: uv (fast, deterministic installs) or pip/venv

Using uv (recommended):
- Install uv: https://github.com/astral-sh/uv
- In the repo root:
  - `uv venv`
  - `uv pip install -e .`

Using pip:
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -e .`

2) Prepare your dataset
Create a directory with paired images. Each image in `original/` must have a file with the exact same name in `starless/`.
```
Project root
└── data/
    ├── original/
    │   ├── image_0001.tif
    │   └── image_0002.png
    └── starless/
        ├── image_0001.tif
        └── image_0002.png
```
Supported formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`. Images can be 8‑bit, 16‑bit, or float; they are normalized to [0, 1]. Grayscale inputs are broadcast to 3 channels automatically. Images smaller than the configured patch size are skipped.

3) Train
```sh
python main.py
```
This builds the Orion model, creates a dataset of random 256×256 patches from your images, trains with the configured loss using Optax AdamW + warmup‑cosine schedule, and periodically logs metrics and sample predictions.

Override anything on the command line (see `python main.py --help`):
```sh
python main.py --name baseline --epochs 50 --lr 2e-4 --no-show
python main.py --name deeper --bottleneck-depth 4
```

Each run writes everything under `<runs_root>/<name>/` (default `./runs/<name>/`):
- `config.json` — resolved configuration snapshot
- `history.json` — per‑step train metrics + per‑epoch validation metrics (rewritten every epoch, so a killed run still has data)
- `curves.png` — last training‑curves figure
- `sample_epoch_NNN.png` — sample predictions
- `latest/`, `best/` — Orbax checkpoints (resume is automatic unless `--no-resume` is passed)

4) Compare runs
```sh
python compare_runs.py runs/baseline runs/struct          # specific runs
python compare_runs.py --runs-root runs --pattern '*'     # all runs
python compare_runs.py runs/baseline --out cmp.png        # save figure, no display
```
Overlays validation loss, loss components, and the learning‑rate schedule; prints a final summary table.


## Model overview

The network (see `network.py`):
- Encoder: stack of `NAFBlock` (NAFNet residual block — LayerNorm → conv → SimpleGate → SimpleChannelAttention) at 64/128/256/512 channels, with 2×2 average‑pooling between levels
- Bottleneck: `HybridBottleneck` — alternating `HybridLayer`s that mix a `ViTLayer` (MHSA + Swish‑MLP, with attention/MLP dropout) and a `NAFBlock`, plus learnable 2D positional embeddings and DropPath. Surrounded by a global residual.
- Decoder: `UpsampleBlock` (conv → PixelShuffle → Swish), skip connections passed through an `SEGate` (Squeeze‑Excitation), concatenated, then another `NAFBlock`
- Output: a 1×1 convolution predicts a residual that is subtracted from the input (`input − residual`)


## Configuration

`config.py` is the single source of defaults; command‑line flags override them (see `main.py --help`). Defaults:

| Key | Default | Notes |
|---|---|---|
| `patch_size` | 256 | random crop size |
| `batch_size` | 8 | fits a 12 GB GPU in bf16; raise on larger cards |
| `steps_per_epoch` | 250 | |
| `val_steps` | 50 | |
| `val_split` | 0.1 | image‑level train/val split (last 10% held out) when `val_data_dir == data_dir` |
| `augmentation_prob` | 0.5 | per‑patch probability of photometric (gamma/brightness) jitter; D4 orientation aug is always on for train |
| `epochs` | 10 | |
| `lr` / `starting_lr` | 5e‑4 / 1e‑6 | peak and floor of the warmup‑cosine schedule |
| `warmup_epochs` | 1 | |
| `bottleneck_depth` | 2 | number of `HybridLayer`s in the bottleneck |
| `naf_expansion` | 2 | hidden‑channel multiplier in `NAFBlock` |
| `vit_mlp_dropout_rate` | 0.1 | dropout in ViT MLP |
| `stochastic_depth_rate` | 0.05 | DropPath in bottleneck (linearly scaled per layer) |
| `conv_dropout_rate` | 0.05 | dropout in `NAFBlock`s |
| `loss_weights` | see below | per‑component weights for the loss (a subset of the named terms) |
| `precision` | `"bf16"` | `"bf16"` (mixed‑precision compute, faster, less VRAM) or `"fp32"` |
| `grad_clip` | 1.0 | global gradient‑norm clip (set to `None` to disable) |
| `eval_every` | 2 | log train metrics every N steps |
| `visualize_every` / `plot_every` | 5 / 5 | training‑curves / sample‑prediction period (epochs) |
| `epoch_checkpoints` | 5 | |
| `runs_root` / `run_name` | `./runs` / `default` | output goes to `runs_root/run_name/` |

Loss (`loss_functions.py`) — a weighted sum over any of these terms (set in `loss_weights`; only named terms are computed):
- `charbonnier` (1.0) — robust pixel reconstruction (linear).
- `log_charbonnier` (1.0) — Charbonnier in `log(img+floor)` space. Compresses bright stars and amplifies faint ones, so dim/low‑contrast stars get more gradient weight (targets the red‑star‑on‑nebula problem).
- `ssim` (1.0) — 1 − SSIM, structural similarity.
- `gradient` (1.0) — L1 on spatial gradients; penalises residual star edges.
- `frequency` (0.3) — L1 on the log‑magnitude FFT (low/high bands); penalises residual point‑source energy.

Defaults are `{"charbonnier": 1.0, "log_charbonnier": 1.0, "ssim": 1.0, "gradient": 1.0, "frequency": 0.3}`. The components have different natural scales (frequency is large, charbonnier small), so check the per‑component curves in `curves.png` and rebalance weights so no single term dominates. Drop terms you don't want (e.g. `frequency`) by removing the key.


## Data pipeline details

`dataset.py` is pure numpy/cv2:
- All compatible pairs are loaded into host memory once as float32 in [0, 1]
- Each step samples an image (area‑weighted), a random crop, and optional augmentation
- Augmentation: a uniformly random D4 orientation (all 8 dihedral orientations) is always applied during training; `augmentation_prob` additionally controls per‑patch gamma/brightness photometric jitter (applied identically to input and target, so the mapping stays consistent)
- A background thread double‑buffers batches (`prefetch=2`) so the GPU is never starved
- Validation uses a deterministic seed per dataset, so val metrics are comparable across epochs and runs


## Notes, tips, and troubleshooting
- Mixed precision: training defaults to bf16 compute (`precision: "bf16"`) for speed and lower VRAM; pass `--precision fp32` for full precision. Some parameters (e.g. LayerNorm) are kept in float32 automatically.
- Learning rate: warmup‑cosine decay over `epochs * steps_per_epoch`, peaking at `lr` after `warmup_epochs`, decaying back to `starting_lr`. Requires `epochs > warmup_epochs`.
- Checkpoint resume: on by default. Use a unique `--name` per configuration when comparing, or pass `--no-resume` to start fresh into the same dir.
- Dataset quality: for best results provide carefully aligned pairs (original vs starless) with identical filenames and shapes.

Common errors
- `ValueError: "No compatible images found"` / `"No valid image pairs loaded"` → check your data folder structure and file extensions.
- `Shape mismatch for <file>` → ensure originals and starless images have identical width/height and the same filename.
- `Image too small for patch size` → reduce `patch_size` (or `--patch-size`) or remove tiny images.
- `cosine_decay_schedule requires positive decay_steps` → `epochs == warmup_epochs`; bump epochs or shorten warmup.


## Development
- The code uses Flax `nnx` APIs which differ from `nn.Module`/linen: parameters live on the module instances and are updated via `nnx.Optimizer`.
- Module layout:
  - `config.py` — default config + validation
  - `cli_args.py` — `--name`/`--epochs`/... argument parsing, applied on top of `get_default_config()`
  - `dataset.py` — JAX‑native patch data pipeline
  - `network.py` — `Orion` model (NAFBlock, HybridBottleneck, SEGate, etc.)
  - `loss_functions.py` — Charbonnier + SSIM loss
  - `train.py` — optimizer, metrics, jitted `train_step` / `val_step`
  - `checkpoint.py` — Orbax‑based save/restore + resume
  - `metrics_utils.py` — metrics formatting helper
  - `utils.py` — plotting
  - `main.py` — training entry point
  - `compare_runs.py` — side‑by‑side comparison of run histories


## Citation
If you use this in academic work, please cite the repository and the Coordinate Attention paper:
- Repo: "Orion JAX — A Neural Star Remover"
- [Qibin Hou, Daquan Zhou, Jiashi Feng, Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907).


## Acknowledgements
- Thanks to Mikita (Nikita) Misiura for StarNet
- Thanks to the JAX, Flax, and Optax teams for excellent tooling.
