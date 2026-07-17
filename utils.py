import matplotlib
matplotlib.use('Agg')  # Headless backend: plt.show() becomes a no-op so the
                       # training loop never blocks on a GUI window. Figures are
                       # still saved to disk via save_path.
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_images(orig, starless, output, epoch, show: bool = True,
                save_path: Optional[str] = None):
    """Plot the original, starless, and output images side by side.
    
    Args:
        orig: Original input image.
        starless: Ground truth starless image.
        output: Model prediction.
        epoch: Current epoch number.
        show: If True, displays the plot interactively.
        save_path: If given, saves the figure to this path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Clip values to [0, 1] for display
    orig = np.clip(orig, 0, 1)
    starless = np.clip(starless, 0, 1)
    output = np.clip(output, 0, 1)

    # Plot images
    axes[0].imshow(orig)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(starless)
    axes[1].set_title("Starless Target")
    axes[1].axis("off")

    axes[2].imshow(output)
    axes[2].set_title("Network Output")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_training_curves(metrics_history: Dict[str, List[float]], show: bool = True,
                         save_path: Optional[str] = None):
    """Plot training and validation curves for whatever metrics are present.

    Works generically for both the legacy (l1/l2) and enhanced (charbonnier,
    ssim, ...) loss setups: it discovers every ``train_<name>``/``val_<name>``
    pair and plots them alongside total loss and the learning-rate schedule.

    Args:
        metrics_history: Dictionary containing training metrics over time.
        show: If True, displays the plot interactively.
        save_path: If given, saves the figure to this path.
    """
    if len(metrics_history.get('val_loss', [])) == 0:
        print("No validation metrics to plot yet")
        return

    train_steps = max(1, len(metrics_history['train_loss']))
    val_points = max(1, len(metrics_history['val_loss']))
    val_epochs_scaled = [
        e * (train_steps / val_points)
        for e in range(len(metrics_history['val_loss']))
    ]

    # Discover component pairs (anything with both train_ and val_ entries),
    # excluding the aggregate 'loss' which is plotted separately.
    components = []
    for k in metrics_history:
        if k.startswith('train_') and k != 'train_loss':
            name = k[len('train_'):]
            if f'val_{name}' in metrics_history:
                components.append(name)
    components = sorted(set(components))

    n_plots = 1 + len(components) + 1  # total loss + components + lr
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    def plot_pair(ax, key: str, title: str) -> None:
        ax.set_title(title)
        if len(metrics_history[f'train_{key}']) > 0:
            ax.semilogy(metrics_history[f'train_{key}'], label=f'train_{key}', alpha=0.7)
        ax.semilogy(val_epochs_scaled, metrics_history[f'val_{key}'],
                    label=f'val_{key}', marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    idx = 0
    plot_pair(axes[idx], 'loss', 'Total Weighted Loss')
    idx += 1
    for c in components:
        plot_pair(axes[idx], c, c.replace('_', ' ').title())
        idx += 1

    ax = axes[idx]
    ax.set_title('Learning Rate Schedule')
    ax.plot(metrics_history['learning_rate'], label='learning_rate', color='green')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    idx += 1

    for j in range(idx, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig