import random

import cv2
from functools import partial
from utils import plot_images
from dataset import create_dataset
from network import Orion
from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time
import matplotlib.pyplot as plt


# training configuration
config = {
    "patch_size": 256,
    "batch_size": 16,
    "steps_per_epoch": 250,
    "lr": 1e-4,
    "starting_lr": 1e-6,
    "epochs": 10,
    "warmup_epochs": 1,
    "precision": "bf16",
    "augmentation_prob": 0.5,
    "alpha": 100, # L1 weight
    "beta": 100, # L2 weight
    "T_0": 20, # decay length
    "T_mult": 2,
    "epoch_visualization": 10,
    "epoch_checkpoints": 20,
    "data_dir": "./data",
    "shuffle_buffer_size": 1000
}

def main():
    print("Initializing dataset...")
    dataset = create_dataset(config)

    print("Initializing model...")
    model = Orion(in_channels=3, bottleneck_depth=2, rngs=nnx.Rngs(0))

    # TODO add gradient accumulation

    # print("Warming up...")
    # model(jnp.ones((config['batch_size'], 256, 256, 3)))

    print("Initializing optimizer...")
    learning_rate = 0.005
    momentum = 0.9

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        l1_loss=nnx.metrics.Average('l1_loss'),
        l2_loss=nnx.metrics.Average('l2_loss'),
    )

    def loss_fn(model: model, batch, l1_weight: float, l2_weight: float):
        inputs, targets = batch
        predictions = model(inputs)
        l1_loss = jnp.abs(predictions - targets).mean()
        l2_loss = jnp.square(predictions - targets).mean()
        loss = l1_weight * l1_loss + l2_weight * l2_loss
        return loss, (l1_loss, l2_loss)

    @nnx.jit
    def train_step(model: model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, l1_weight: float, l2_weight: float):
        """Train for a single step."""
        _loss_fn = lambda m, b: loss_fn(m, b, l1_weight=l1_weight, l2_weight=l2_weight)
        grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
        (total_loss, (l1_loss, l2_loss)), grads = grad_fn(model, batch)
        metrics.update(loss=total_loss, l1_loss=l1_loss, l2_loss=l2_loss)  # In-place updates.
        optimizer.update(grads)  # In-place updates.

    @nnx.jit
    def eval_step(model: model, metrics: nnx.MultiMetric, batch, l1_weight: float, l2_weight: float):
        total_loss, (l1_loss, l2_loss) = loss_fn(model, batch, l1_weight=l1_weight, l2_weight=l2_weight)
        metrics.update(loss=total_loss, l1_loss=l1_loss, l2_loss=l2_loss)

    eval_every = 2
    visualize_every = 5
    plot_every = 5
    train_steps = 1000
    l1_loss_weight = 2.
    l2_loss_weight = 8.

    metrics_history = {
      'train_loss': [],
      'train_l1_loss': [],
      'train_l2_loss': [],
    }

    for epoch in range(config['epochs']):
        pbar = tqdm(total=config['steps_per_epoch'], desc=f'Epoch {epoch+1}')
        for step, batch in enumerate(dataset.as_numpy_iterator()):
            train_step(model, optimizer, metrics, batch, l1_weight=l1_loss_weight, l2_weight=l2_loss_weight)
            if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
                # Log training metrics
                computed_train_metrics = metrics.compute()
                for metric_name, value in computed_train_metrics.items():
                    history_key = f'train_{metric_name}'
                    if history_key in metrics_history:
                        metrics_history[history_key].append(value)
                    else:
                        print(f"Warning: Metric '{metric_name}' computed but not tracked in metrics_history.")
                metrics.reset()  # Reset metrics before evaluation
            pbar.update(1)
        pbar.close()
        if epoch % visualize_every == 0:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))  # Adjusted for 3 plots
            metrics_history['train_loss'].append(metrics_history['train_loss'][-1])
            # Plot Total Loss
            ax1.set_title('Total Weighted Loss')
            ax1.semilogy(metrics_history['train_loss'], label='train_loss')
            ax1.set_xlabel('Evaluation Step')
            ax1.set_ylabel('Loss')
            ax1.legend()

            # Plot L1 Loss
            ax2.set_title('L1 Loss (MAE)')
            ax2.semilogy(metrics_history['train_l1_loss'], label='train_l1_loss')
            ax2.set_xlabel('Evaluation Step')
            ax2.set_ylabel('L1 Loss')
            ax2.legend()

            # Plot L2 Loss
            ax3.set_title('L2 Loss (MSE)')
            ax3.semilogy(metrics_history['train_l2_loss'], label='train_l2_loss')
            ax3.set_xlabel('Evaluation Step')
            ax3.set_ylabel('L2 Loss')
            ax3.legend()

            plt.tight_layout()
            plt.show()
        if epoch % plot_every == 0:
            batch = next(dataset.as_numpy_iterator())
            orig, starless = batch
            plot_images(orig[0], starless[0], model(orig[0][None, :])[0], epoch)


if __name__ == "__main__":
    main()
