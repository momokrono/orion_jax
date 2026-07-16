"""Loss functions for the Orion star removal network.

The combined loss is a weighted sum over any subset of:

- ``charbonnier``       — robust pixel reconstruction (L1-like, smooth near 0)
- ``log_charbonnier``   — same, in ``log(img + floor)`` space. Compresses bright
                          stars and relatively amplifies faint/low-contrast
                          features, so dim stars (e.g. red on an orange nebula)
                          get more gradient weight — directly targets the
                          contrast-sensitivity of plain pixel losses.
- ``ssim``              — 1 - SSIM, structural similarity (windowed)
- ``gradient``          — L1 on spatial gradients; penalises residual star edges
- ``frequency``         — L1 on log-magnitude FFT, split into low/high bands;
                          penalises residual point-source (high-freq) energy

Only the components named as keys in the ``weights`` dict are computed.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Callable


def charbonnier_loss(pred: jnp.ndarray,
                     target: jnp.ndarray,
                     epsilon: float = 1e-3) -> jnp.ndarray:
    """Charbonnier loss: mean(sqrt((pred - target)^2 + epsilon^2))."""
    diff = pred - target
    return jnp.mean(jnp.sqrt(diff ** 2 + epsilon ** 2))


def log_charbonnier_loss(pred: jnp.ndarray,
                         target: jnp.ndarray,
                         floor: float = 1e-3,
                         epsilon: float = 1e-3) -> jnp.ndarray:
    """Charbonnier loss computed in log-space.

    ``floor`` keeps the log argument strictly positive (and sets the dark-region
    behaviour). Operating in log-space compresses the huge astro dynamic range,
    so faint/low-contrast stars receive relatively more loss weight than in
    linear space.
    """
    log_pred = jnp.log(jnp.clip(pred, 0.0, None) + floor)
    log_target = jnp.log(jnp.clip(target, 0.0, None) + floor)
    diff = log_pred - log_target
    return jnp.mean(jnp.sqrt(diff ** 2 + epsilon ** 2))


def ssim_loss(pred: jnp.ndarray,
              target: jnp.ndarray,
              max_val: float = 1.0,
              filter_size: int = 11,
              k1: float = 0.01,
              k2: float = 0.03) -> jnp.ndarray:
    """SSIM loss (1 - SSIM) with an 11x11 Gaussian window, per-channel."""
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    def gaussian_kernel_1d(size: int, sigma: float) -> jnp.ndarray:
        coords = jnp.arange(size, dtype=jnp.float32)
        coords -= (size - 1) / 2.0
        g = jnp.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / jnp.sum(g)

    kernel_1d = gaussian_kernel_1d(filter_size, 1.5)
    kernel = jnp.outer(kernel_1d, kernel_1d)[:, :, None, None]  # [H, W, 1, 1]

    def apply_filter(img):
        N, H, W, C = img.shape
        kernel_hwio = jnp.broadcast_to(kernel, (kernel.shape[0], kernel.shape[1], 1, C))
        return jax.lax.conv_general_dilated(
            img, kernel_hwio, window_strides=(1, 1), padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=C,
        )

    mu_pred = apply_filter(pred)
    mu_target = apply_filter(target)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = apply_filter(pred ** 2) - mu_pred_sq
    sigma_target_sq = apply_filter(target ** 2) - mu_target_sq
    sigma_pred_target = apply_filter(pred * target) - mu_pred_target

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    return 1.0 - jnp.mean(numerator / (denominator + 1e-8))


def gradient_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """L1 on first-order spatial gradients (x and y), averaged."""
    pred_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    pred_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, 1:, :, :] - target[:, :-1, :, :]
    target_dx = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_dy = jnp.abs(pred_dy - target_dy).mean()
    loss_dx = jnp.abs(pred_dx - target_dx).mean()
    return (loss_dy + loss_dx) / 2.0


def frequency_loss(pred: jnp.ndarray,
                   target: jnp.ndarray,
                   alpha: float = 0.5) -> jnp.ndarray:
    """Frequency-domain loss on the grayscale log-magnitude spectrum.

    Splits the spectrum into low (nebula-scale) and high (star/point-source)
    bands and penalises residual energy in each. ``alpha`` balances the bands
    (0 = low only, 1 = high only). More contrast-robust than pixel losses for
    detecting leftover point sources.
    """
    pred_gray = jnp.mean(pred, axis=-1)            # [N, H, W]
    target_gray = jnp.mean(target, axis=-1)

    pred_mag = jnp.log1p(jnp.abs(jnp.fft.fft2(pred_gray)))
    target_mag = jnp.log1p(jnp.abs(jnp.fft.fft2(target_gray)))

    N, H, W = pred_mag.shape
    y = jnp.arange(H)[:, None] - H // 2
    x = jnp.arange(W)[None, :] - W // 2
    radius = jnp.sqrt(y ** 2 + x ** 2)

    low_mask = radius < (min(H, W) * 0.2)
    high_mask = ~low_mask

    low_diff = jnp.abs((pred_mag - target_mag) * low_mask)
    high_diff = jnp.abs((pred_mag - target_mag) * high_mask)

    low_loss = jnp.sum(low_diff) / (jnp.sum(low_mask) + 1e-8)
    high_loss = jnp.sum(high_diff) / (jnp.sum(high_mask) + 1e-8)
    return (1 - alpha) * low_loss + alpha * high_loss


# Dispatch table: loss name -> function. combined_loss computes whichever keys
# appear in the weights dict.
LOSS_FUNCTIONS: Dict[str, Callable] = {
    'charbonnier': charbonnier_loss,
    'log_charbonnier': log_charbonnier_loss,
    'ssim': ssim_loss,
    'gradient': gradient_loss,
    'frequency': frequency_loss,
}


def combined_loss(pred: jnp.ndarray,
                  target: jnp.ndarray,
                  weights: Dict[str, float]) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Weighted sum over the loss components named in ``weights``.

    Args:
        pred: Predicted starless image [N, H, W, C].
        target: Ground truth starless image [N, H, W, C].
        weights: Dict mapping component name (a key of ``LOSS_FUNCTIONS``) to
            its weight. Unknown names are ignored.

    Returns:
        Tuple of (total_loss, loss_components_dict).
    """
    loss_components: Dict[str, jnp.ndarray] = {}
    for name in weights:
        fn = LOSS_FUNCTIONS.get(name)
        if fn is not None:
            loss_components[name] = fn(pred, target)

    total_loss = sum(weights[name] * value for name, value in loss_components.items())
    return total_loss, loss_components
