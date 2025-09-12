import matplotlib.pyplot as plt
import numpy as np


def plot_images(orig, starless, output, epoch, save: bool = False):
    """Plot the original, starless, and output images side by side."""
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

    # Save the figure
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(f"epoch_{epoch + 1}_comparison.png")
    plt.close()