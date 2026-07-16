"""Training utilities and step functions for Orion."""

from typing import Dict, Any, Tuple
import jax.numpy as jnp
from flax import nnx
import optax
from loss_functions import combined_loss


def create_optimizer(model: nnx.Module, config: Dict[str, Any],
                     lr_schedule: optax.Schedule = None) -> nnx.Optimizer:
    """Create the AdamW optimizer with gradient clipping and an LR schedule.

    Args:
        model: The neural network model.
        config: Training configuration.
        lr_schedule: Optional pre-built LR schedule. If None, it is built from config.

    Returns:
        Configured nnx.Optimizer.
    """
    if lr_schedule is None:
        warmup_steps = config['warmup_epochs'] * config['steps_per_epoch']
        total_steps = config['epochs'] * config['steps_per_epoch']
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config['starting_lr'],
            peak_value=config['lr'],
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=config['starting_lr'],
        )

    momentum = config.get('momentum', 0.9)
    base_optimizer = optax.adamw(lr_schedule, momentum)

    grad_clip = config.get('grad_clip')
    if grad_clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip), base_optimizer)
    else:
        tx = base_optimizer

    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def create_metrics(component_names) -> nnx.MultiMetric:
    """Create a metrics tracker for 'loss' and each loss component name."""
    kwargs = {'loss': nnx.metrics.Average('loss')}
    for name in component_names:
        kwargs[name] = nnx.metrics.Average(name)
    return nnx.MultiMetric(**kwargs)


def loss_fn(model: nnx.Module,
            batch: Tuple[jnp.ndarray, jnp.ndarray],
            loss_weights: Dict[str, float],
            training: bool = True) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute the combined Charbonnier + SSIM loss for a batch.

    Args:
        model: The neural network model.
        batch: Tuple of (inputs, targets).
        loss_weights: Dictionary of weights for each loss component.
        training: If True, runs the model in training mode (enables dropout).

    Returns:
        Tuple of (total_loss, loss_components_dict).
    """
    inputs, targets = batch
    predictions = model(inputs, run=training)
    total_loss, loss_components = combined_loss(predictions, targets, loss_weights)
    return total_loss, loss_components


@nnx.jit
def train_step(model: nnx.Module,
               optimizer: nnx.Optimizer,
               metrics: nnx.MultiMetric,
               batch: Tuple[jnp.ndarray, jnp.ndarray],
               loss_weights: Dict[str, float]) -> None:
    """Perform a single training step (model/optimizer/metrics updated in-place)."""
    def _loss_fn(m, b):
        return loss_fn(m, b, loss_weights=loss_weights, training=True)
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (total_loss, loss_components), grads = grad_fn(model, batch)
    metrics.update(loss=total_loss, **loss_components)
    optimizer.update(model, grads)


@nnx.jit
def val_step(model: nnx.Module,
             metrics: nnx.MultiMetric,
             batch: Tuple[jnp.ndarray, jnp.ndarray],
             loss_weights: Dict[str, float]) -> None:
    """Perform a single validation step (metrics updated in-place)."""
    total_loss, loss_components = loss_fn(
        model, batch, loss_weights=loss_weights, training=False
    )
    metrics.update(loss=total_loss, **loss_components)


def initialize_metrics_history(component_names) -> Dict[str, list]:
    """Initialize the metrics history for loss and each active component."""
    history: Dict[str, list] = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'epochs': [],
    }
    for name in component_names:
        if name == 'loss':
            continue
        history[f'train_{name}'] = []
        history[f'val_{name}'] = []
    return history
