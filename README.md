# Orion JAX — Neural Star Remover (JAX/Flax)

Orion is a neural network that removes stars from deep‑sky astrophotography images, producing a “starless” version that can be used for processing nebulosity and background structures. This repository is a JAX/Flax (nnx) port of an earlier PyTorch implementation.

Highlights
- JAX/Flax (nnx) implementation with jit-compiled training/evaluation steps
- U‑Net‑like encoder–decoder with residual blocks and Coordinate Attention
- PixelShuffle upsampling for efficient, artifact‑free upscaling
- Simple TF‑based input pipeline that reads large TIFF/PNG/JPG pairs and yields random crops


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
Create a directory with paired images. Each image in original/ must have a file with the exact same name in starless/.
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
Supported formats: .tif, .tiff, .png, .jpg, .jpeg. Images can be 8‑bit, 16‑bit, or float; they are normalized to [0, 1]. Grayscale inputs are broadcast to 3 channels automatically. Images smaller than the configured patch size are skipped.

Note: The loader uses TensorFlow only for the data pipeline (CPU). It explicitly disables TF GPU to avoid conflicts with JAX.

3) Train
- Default config is in main.py (see the config dict). Run:

```sh
python main.py
```

This will:
- Build the Orion model
- Create a dataset of random 256×256 patches from your images
- Train with a weighted L1+L2 loss (MAE+MSE) using Optax AdamW
- Periodically log metrics and show example predictions


## Model overview

The network (see network.py):
- Encoder: ResidualBlock stacks with BatchNorm and Swish activations
- Bottleneck: Configurable depth of ResidualBlocks
- Decoder: PixelShuffle upsampling, skip connections gated via Coordinate Attention
- Output: A 1×1 convolution predicts a residual which is subtracted from the input (input − residual)

Key components
- ResidualBlock: Conv → BN → Swish, CoordAttn, with a 1×1 shortcut
- CoordAttn: Coordinate Attention module adapted for NHWC
- UpsampleBlock: Conv → PixelShuffle → Swish


## Configuration
Edit main.py to change training and data parameters. Defaults:
- patch_size: 256
- batch_size: 16
- steps_per_epoch: 250
- lr: 1e-4 (note: optimizer currently uses a fixed value set near initialization)
- starting_lr: 1e-6
- epochs: 10
- warmup_epochs: 1
- precision: "bf16"
- augmentation_prob: 0.5
- alpha: 100  (L1 weight, not used directly in current loss function)
- beta: 100   (L2 weight, not used directly in current loss function)
- T_0: 20, T_mult: 2
- epoch_visualization: 10
- epoch_checkpoints: 20
- data_dir: ./data
- shuffle_buffer_size: 1000

Loss weights used in the loop (can be edited):
- l1_loss_weight = 2.0
- l2_loss_weight = 8.0

Dataset options (dataset.py):
- Random cropping to patch_size
- Optional augmentations (flip/rotate, brightness/contrast, color jitter) with probability augmentation_prob


## Data pipeline details
- Files are loaded into memory once (only pairs that are at least patch_size on both dimensions)
- Each training step draws a random crop from a random image pair
- Normalization: uint8→/255, uint16→/65535, float32/64→as is
- Grayscale images are replicated to RGB
- TensorFlow is used to wrap the generator and batch/shuffle efficiently on CPU


## Notes, tips, and troubleshooting
- GPU memory: TensorFlow is configured to ignore GPUs to prevent it from claiming the device JAX should use.
- Mixed precision: precision is set in config, but compute dtypes are passed per-module. You can experiment with jnp.bfloat16 for speed on supported hardware.
- Learning rate: The optimizer is initialized with a fixed value in main.py (learning_rate = 0.005). Adjust to match config["lr"], or wire a scheduler if desired.
- Visualizations: The script periodically plots train metrics and shows model outputs on a sample patch. You can set save=True in utils.plot_images to save PNGs per epoch.
- Dataset quality: For best results, provide carefully aligned pairs (original vs starless) with identical filenames and shapes.

Common errors
- ValueError: "No compatible images found" or "No valid image pairs loaded" → Check your data folder structure and file extensions.
- Shape mismatch for <file> → Ensure originals and starless images have identical width/height and same filename.
- Image too small for patch size → Reduce config["patch_size"] or remove tiny images.


## Development
- devel.ipynb and test.ipynb contain scratch work and small experiments.
- The code uses Flax nnx APIs which differ from nn.Module/linen; parameters live on the module instances and are updated via nnx.Optimizer.


## Citation
If you use this in academic work, please cite the repository and the Coordinate Attention paper:
- Repo: "Orion JAX — A Neural Star Remover"
- [Qibin Hou, Daquan Zhou, Jiashi Feng, Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907).


## Acknowledgements
- Thanks to Mikita (Nikita) Misiura for StarNet
- Thanks to the JAX, Flax, and Optax teams for excellent tooling.