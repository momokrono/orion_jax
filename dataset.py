"""Jax-native data pipeline for Orion (no TensorFlow).

Images are loaded into host memory once (numpy float32). Each step samples
patches (area-weighted by image size), crops them, and applies augmentation
with numpy/cv2. Batches are yielded as numpy arrays, which JAX transfers to
device inside the jitted train/val steps.
"""

import os
import queue
import threading
import cv2
import numpy as np
import tifffile
from typing import Dict, Any, Iterator, List, Tuple

_IMAGE_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')


def _normalize(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        factor = 255.0
    elif img.dtype == np.uint16:
        factor = 65535.0
    elif img.dtype in (np.float32, np.float64):
        factor = 1.0
    else:
        raise TypeError(f"Unsupported image dtype: {img.dtype}")
    return np.clip(img.astype(np.float32) / factor, 0.0, 1.0)


def _load_one(path: str) -> np.ndarray:
    if path.lower().endswith(('.tif', '.tiff')):
        return tifffile.imread(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _to_three_channels(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def load_images_to_memory(data_dir: str, patch_size: int, split: str = 'all',
                          val_split: float = 0.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Loads original and starless image pairs as float32 numpy arrays.

    When val_split > 0 and split is 'train' or 'val', the sorted image list is
    partitioned deterministically: the last val_split fraction of images is held
    out for validation, the rest used for training. This prevents patch-level
    leakage between train and validation.
    """
    original_dir = os.path.join(data_dir, "original")
    starless_dir = os.path.join(data_dir, "starless")

    try:
        all_files = os.listdir(original_dir)
        image_fnames = sorted(
            f for f in all_files
            if os.path.isfile(os.path.join(original_dir, f)) and f.lower().endswith(_IMAGE_EXTS)
        )
    except FileNotFoundError:
        raise ValueError(f"Original directory not found: {original_dir}")

    if not image_fnames:
        raise ValueError(f"No compatible images found in {original_dir}")

    if val_split > 0.0 and len(image_fnames) >= 2 and split in ('train', 'val'):
        total = len(image_fnames)
        n_val = max(1, round(total * val_split))
        if split == 'train':
            image_fnames = image_fnames[:total - n_val]
        else:
            image_fnames = image_fnames[total - n_val:]
        print(f"[split={split}] using {len(image_fnames)}/{total} image pairs")

    originals: List[np.ndarray] = []
    starlesses: List[np.ndarray] = []
    for fname in image_fnames:
        orig_path = os.path.join(original_dir, fname)
        starless_path = os.path.join(starless_dir, fname)
        if not os.path.exists(starless_path):
            continue
        try:
            orig = _load_one(orig_path)
            star = _load_one(starless_path)
            if orig is None or star is None:
                continue
            orig = _to_three_channels(orig)
            star = _to_three_channels(star)
            if orig.shape != star.shape:
                print(f"Shape mismatch for {fname}. Skipping.")
                continue
            if orig.shape[0] < patch_size or orig.shape[1] < patch_size:
                print(f"{fname} too small ({orig.shape[:2]}) for patch {patch_size}. Skipping.")
                continue
            originals.append(_normalize(orig))
            starlesses.append(_normalize(star))
        except Exception as e:
            print(f"Error loading {fname}: {e}. Skipping.")

    print(f"Loaded {len(originals)} valid image pairs into memory.")
    if not originals:
        raise ValueError("No valid image pairs loaded.")
    return originals, starlesses


class AstroDataset:
    """Iterable patch dataset with numpy/cv2 augmentation.

    Yields infinite batches of (orig, starless) as float32 numpy arrays of shape
    (batch_size, patch_size, patch_size, 3). Training samples are random and
    augmented; validation samples are deterministic (same seed every full pass)
    and unaugmented, so val metrics are comparable across epochs.
    """

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.split = split
        self.patch_size = int(config['patch_size'])
        self.batch_size = int(config['batch_size'])
        self.augmentation_prob = float(config.get('augmentation_prob', 0.0))
        # Global photometric jitter strengths. ``gamma_strength`` is the
        # half-width of a log-uniform stretch (so 0.7 -> 2^U(-0.7,0.7) ~= 0.62..1.62,
        # applied identically to R/G/B). ``gain_strength`` is the half-width of
        # an additive multiplicative gain (0.2 -> U(0.8, 1.2), also global).
        # Both are channel-coupled — see _photometric for the rationale.
        self.gamma_strength = float(config.get('photo_gamma_strength', 0.7))
        self.gain_strength = float(config.get('photo_gain_strength', 0.2))
        self.prefetch = int(config.get('prefetch', 2))
        self.is_validation = (split == 'val')
        # Deterministic, reproducible crops for validation; fresh stream for train.
        self.seed = config.get('seed', 0 if self.is_validation else None)

        self.originals, self.starlesses = load_images_to_memory(
            config['data_dir'], self.patch_size, split=split,
            val_split=config.get('val_split', 0.0),
        )
        areas = np.array([im.shape[0] * im.shape[1] for im in self.originals], dtype=np.float64)
        self.img_probs = areas / areas.sum()

    def _new_rng(self) -> np.random.RandomState:
        return np.random.RandomState(self.seed)

    def _sample_patch(self, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        idx = rng.choice(len(self.originals), p=self.img_probs)
        orig = self.originals[idx]
        star = self.starlesses[idx]
        h, w = orig.shape[0], orig.shape[1]
        p = self.patch_size
        y = rng.randint(0, h - p + 1)
        x = rng.randint(0, w - p + 1)
        return orig[y:y + p, x:x + p], star[y:y + p, x:x + p]

    def _augment(self, orig: np.ndarray, star: np.ndarray,
                 rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric (D4) and photometric augmentation.

        - Geometric: a uniformly random dihedral orientation (all 8 of the D4
          group: 4 rotations x optional flip). Always applied during training
          for maximum orientation diversity. When ``augmentation_prob == 0``
          (validation), nothing is applied.
        - Photometric: gamma + brightness jitter, gated per-patch by
          ``augmentation_prob``. Applied identically to orig and star so the
          input->target mapping stays consistent.
        """
        if self.augmentation_prob <= 0.0:
            return orig, star

        # Geometric: uniform random D4 orientation.
        k = int(rng.randint(0, 4))
        if rng.random() < 0.5:
            orig = np.ascontiguousarray(np.flip(orig, axis=1))
            star = np.ascontiguousarray(np.flip(star, axis=1))
        if k:
            orig = np.ascontiguousarray(np.rot90(orig, k))
            star = np.ascontiguousarray(np.rot90(star, k))

        # Photometric (per-channel gamma/gain) jitter.
        if rng.random() < self.augmentation_prob:
            orig, star = self._photometric(
                orig, star, rng, self.gamma_strength, self.gain_strength,
            )

        return np.clip(orig, 0.0, 1.0), np.clip(star, 0.0, 1.0)

    @staticmethod
    def _photometric(orig: np.ndarray, star: np.ndarray,
                     rng: np.random.RandomState,
                     gamma_strength: float, gain_strength: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
        # Global photometric jitter, applied identically to orig and star
        # (same realization) so the input->target mapping stays consistent.
        #
        # NOTE: gamma and gain are GLOBAL (same value across R/G/B), not
        # per-channel. Astrophotography colour is semantically meaningful —
        # H-alpha is red, O-III blue-green, S-II deep red — and per-channel
        # gamma was found to hurt held-out quality (~-2.3 dB PSNR) by
        # distorting the spectral signature the model uses to tell emission
        # nebulosity apart from stars. Keep the photometric transform channel-
        # coupled unless you have a reason to break spectral cues.
        #
        # gamma: log-uniform stretch, 2^U(-g, g).
        # gain: additive multiplicative gain, 1 + U(-a, a).
        gamma = float(2.0 ** rng.uniform(-gamma_strength, gamma_strength))
        gain = float(1.0 + rng.uniform(-gain_strength, gain_strength))
        orig = np.clip((orig ** gamma) * gain, 0.0, 1.0).astype(np.float32)
        star = np.clip((star ** gamma) * gain, 0.0, 1.0).astype(np.float32)
        return orig, star

    def _build_batch(self, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        p = self.patch_size
        origs = np.empty((self.batch_size, p, p, 3), dtype=np.float32)
        stars = np.empty((self.batch_size, p, p, 3), dtype=np.float32)
        for b in range(self.batch_size):
            o, s = self._sample_patch(rng)
            if not self.is_validation:
                o, s = self._augment(o, s, rng)
            origs[b] = o
            stars[b] = s
        return origs, stars

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        rng = self._new_rng()
        if self.prefetch <= 1:
            while True:
                yield self._build_batch(rng)
            return
        yield from self._prefetched(rng)

    def _prefetched(self, rng: np.random.RandomState) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield batches from a background producer thread (double-buffered).

        The producer prepares upcoming batches on the host while the consumer
        trains on the GPU, hiding crop/augment latency. The worker is stopped
        and drained when this generator is closed (e.g. the training loop breaks
        at steps_per_epoch).
        """
        buf: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        sentinel = object()
        errors: List[BaseException] = []

        def producer():
            try:
                while not stop.is_set():
                    buf.put(self._build_batch(rng))
            except BaseException as e:  # surface producer failures to the consumer
                errors.append(e)
                buf.put(sentinel)

        worker = threading.Thread(target=producer, daemon=True)
        worker.start()
        try:
            while True:
                item = buf.get()
                if item is sentinel:
                    if errors:
                        raise errors[0]
                    return
                yield item
        finally:
            stop.set()
            try:
                while True:
                    buf.get_nowait()
            except queue.Empty:
                pass


def create_dataset(config: Dict[str, Any]) -> AstroDataset:
    """Create the iterable patch dataset for the given config.

    Reads ``config['split']`` ('train' or 'val') to select the image partition.
    """
    print("--- Creating Dataset ---")
    split = config.get('split', 'train')
    return AstroDataset(config, split=split)
