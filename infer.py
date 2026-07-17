"""Run a trained Orion model on a full image using overlap-tile inference.

The network only accepts fixed ``patch_size`` inputs (its bottleneck has
learned positional embeddings), so large images are split into overlapping
patches, processed in batches, and blended back together with a linear ramp
over the overlap region (seam-free).

Usage:
    python infer.py runs/run2-loss my_image.tif
    python infer.py runs/run2-loss my_image.tif -o starless.tif --which best
    python infer.py runs/run2-loss img.tif --overlap 48 --batch-size 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from flax import nnx

from network import Orion
from checkpoint import CheckpointManager
from dataset import _load_one, _to_three_channels, _normalize


@nnx.jit
def _predict(model, x):
    """Jitted batched forward in eval mode (no dropout)."""
    return model(x, run=False)


def _starts(length: int, patch: int, stride: int) -> list:
    """Patch start positions covering [0, length); last patch flush to the end."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _blend_vec(start: int, patch: int, overlap: int, length: int) -> np.ndarray:
    """1D blend weight for one patch on an axis.

    Ramps 0->1 over the overlap at each edge that has a neighbouring patch;
    edges flush with the image border stay at 1 (no darkening at the border).
    """
    v = np.ones(patch, dtype=np.float32)
    if start > 0:
        v[:overlap] = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    if start + patch < length:
        v[patch - overlap:] = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
    return v


def tiled_infer(model, img: np.ndarray, patch: int, overlap: int,
                batch_size: int) -> np.ndarray:
    """Run the model over a full image with overlap-tile blending.

    Args:
        img: [H, W, 3] float32 in [0, 1].
    Returns:
        [H, W, 3] float32 in ~[0, 1].
    """
    H, W, _ = img.shape
    pad_h, pad_w = max(0, patch - H), max(0, patch - W)
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    Hp, Wp, _ = img.shape
    stride = patch - overlap

    ys = _starts(Hp, patch, stride)
    xs = _starts(Wp, patch, stride)
    yvecs = [_blend_vec(s, patch, overlap, Hp) for s in ys]
    xvecs = [_blend_vec(s, patch, overlap, Wp) for s in xs]

    tasks = []
    for y0, wy in zip(ys, yvecs):
        for x0, wx in zip(xs, xvecs):
            tasks.append((y0, x0, np.outer(wy, wx).astype(np.float32)))
    n = len(tasks)
    print(f"  tiles: {len(ys)}x{len(xs)} = {n}, patch={patch}, overlap={overlap}")

    canvas = np.zeros((Hp, Wp, 3), dtype=np.float32)
    wsum = np.zeros((Hp, Wp, 1), dtype=np.float32)

    for i in range(0, n, batch_size):
        chunk = tasks[i:i + batch_size]
        actual = len(chunk)
        batch = np.empty((batch_size, patch, patch, 3), dtype=np.float32)
        for j, (y0, x0, _) in enumerate(chunk):
            batch[j] = img[y0:y0 + patch, x0:x0 + patch]
        batch[actual:] = batch[actual - 1]  # pad tail with last real patch

        outs = np.asarray(_predict(model, jnp.asarray(batch)))
        for j in range(actual):
            y0, x0, w2d = chunk[j]
            slab = slice(y0, y0 + patch), slice(x0, x0 + patch)
            canvas[slab] += outs[j] * w2d[..., None]
            wsum[slab] += w2d[..., None]

    result = canvas / np.maximum(wsum, 1e-8)
    return result[:H, :W]


def _save(arr01: np.ndarray, path: str, dtype) -> None:
    """Denormalize [0,1] float32 back to the original dtype and save."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(arr01 * info.max, 0, info.max).astype(dtype)
    else:
        out = arr01.astype(dtype)

    if path.lower().endswith(('.tif', '.tiff')):
        import tifffile
        tifffile.imwrite(path, out)
    else:
        import cv2
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"Saved -> {path}  ({out.dtype}, shape {out.shape})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', help="Run directory (contains config.json + best/ + latest/).")
    p.add_argument('input', help="Input image to process.")
    p.add_argument('-o', '--output', default=None,
                   help="Output path (default: <input>_starless<ext>).")
    p.add_argument('--which', default='best', choices=['best', 'latest'],
                   help="Which checkpoint to load (default: best).")
    p.add_argument('--overlap', type=int, default=32,
                   help="Tile overlap in pixels (default: 32).")
    p.add_argument('--batch-size', type=int, default=8,
                   help="Patch batch size for inference (default: 8).")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        raise SystemExit(f"No config.json in run dir: {run_dir}")
    config = json.loads(config_path.read_text())

    patch = int(config['patch_size'])
    compute_dtype = jnp.bfloat16 if config.get('precision', 'fp32') == 'bf16' else jnp.float32
    print(f"Loading model from {run_dir} [{args.which}] (patch={patch}, "
          f"precision={config.get('precision', 'fp32')})")

    model = Orion(
        in_channels=config['in_channels'],
        bottleneck_depth=config['bottleneck_depth'],
        compute_dtype=compute_dtype,
        rngs=nnx.Rngs(0),
        vit_mlp_dropout_rate=config.get('vit_mlp_dropout_rate', 0.1),
        attn_dropout_rate=config.get('attn_dropout_rate', 0.0),
        stochastic_depth_rate=config.get('stochastic_depth_rate', 0.05),
        conv_dropout_rate=config.get('conv_dropout_rate', 0.05),
        naf_expansion=config.get('naf_expansion', 2),
        input_size=patch,
    )
    CheckpointManager(str(run_dir)).restore_model(model, args.which)

    raw = _load_one(args.input)
    if raw is None:
        raise SystemExit(f"Could not read image: {args.input}")
    orig_dtype = raw.dtype
    img = _normalize(_to_three_channels(raw))
    print(f"Input: {args.input}  ({orig_dtype}, shape {raw.shape})")

    out = tiled_infer(model, img, patch, args.overlap, args.batch_size)

    out_path = args.output
    if out_path is None:
        src = Path(args.input)
        out_path = str(src.with_name(f"{src.stem}_starless{src.suffix}"))
    _save(out, out_path, orig_dtype)


if __name__ == "__main__":
    main()
