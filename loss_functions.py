"""Loss functions for the Orion star removal network.

A single combined loss is used: Charbonnier (robust pixel reconstruction) +
SSIM (structural similarity). Both terms are cheap and complementary for
star-removal: Charbonnier drives pixel-accurate reconstruction while SSIM
preserves large-scale nebular structure.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict


def charbonnier_loss(pred: jnp.ndarray,
                     target: jnp.ndarray,
                     epsilon: float = 1e-3) -> jnp.ndarray:
    """Charbonnier loss (smooth L1).

    More robust to outliers than L2, smoother than L1 near zero.
    Formula: mean(sqrt((pred - target)^2 + epsilon^2)).
    """
    diff = pred - target
    loss = jnp.sqrt(diff ** 2 + epsilon ** 2)
    return jnp.mean(loss)


def ssim_loss(pred: jnp.ndarray,
              target: jnp.ndarray,
              max_val: float = 1.0,
              filter_size: int = 11,
              k1: float = 0.01,
              k2: float = 0.03) -> jnp.ndarray:
    """SSIM loss (1 - SSIM) for minimization.

    Structural similarity preserves nebula and galaxy structures during
    star removal. Uses an 11x11 Gaussian window applied per-channel.
    """
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    def gaussian_kernel_1d(size: int, sigma: float) -> jnp.ndarray:
        coords = jnp.arange(size, dtype=jnp.float32)
        coords -= (size - 1) / 2.0
        g = jnp.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / jnp.sum(g)

    kernel_1d = gaussian_kernel_1d(filter_size, 1.5)
    kernel = jnp.outer(kernel_1d, kernel_1d)
    kernel = kernel[:, :, None, None]  # [H, W, 1, 1]

    def apply_filter(img, kernel):
        """Depthwise convolution for each channel (NHWC)."""
        N, H, W, C = img.shape
        kernel_hwio = jnp.broadcast_to(kernel, (kernel.shape[0], kernel.shape[1], 1, C))
        return jax.lax.conv_general_dilated(
            img,
            kernel_hwio,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=C,
        )

    mu_pred = apply_filter(pred, kernel)
    mu_target = apply_filter(target, kernel)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = apply_filter(pred ** 2, kernel) - mu_pred_sq
    sigma_target_sq = apply_filter(target ** 2, kernel) - mu_target_sq
    sigma_pred_target = apply_filter(pred * target, kernel) - mu_pred_target

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

    ssim_map = numerator / (denominator + 1e-8)
    return 1.0 - jnp.mean(ssim_map)


def combined_loss(pred: jnp.ndarray,
                  target: jnp.ndarray,
                  weights: Dict[str, float]) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Weighted Charbonnier + SSIM loss.

    Args:
        pred: Predicted starless image [N, H, W, C].
        target: Ground truth starless image [N, H, W, C].
        weights: Dict with optional 'charbonnier' and 'ssim' weights.
            Missing keys default to a weight of 1.0.

    Returns:
        Tuple of (total_loss, loss_components_dict).
    """
    loss_components: Dict[str, jnp.ndarray] = {
        'charbonnier': charbonnier_loss(pred, target),
        'ssim': ssim_loss(pred, target),
    }
    total_loss = (
        weights.get('charbonnier', 1.0) * loss_components['charbonnier']
        + weights.get('ssim', 1.0) * loss_components['ssim']
    )
    return total_loss, loss_components
