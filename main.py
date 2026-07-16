"""Main training script for the Orion star removal network.

Usage:
    python main.py                            # default config from config.py
    python main.py --name baseline --epochs 50
    python main.py --name exp --no-show       # headless

Each run writes to ``<runs_root>/<name>/`` (default ``./runs/<name>/``):
    - ``config.json``       : resolved config snapshot
    - ``history.json``      : per-step/per-epoch metrics (updated every epoch)
    - ``latest/``, ``best/``: orbax checkpoints
    - ``curves.png``        : last training-curves figure
    - ``sample_epoch_N.png``: sample predictions

Compare runs afterwards with ``python compare_runs.py runs/baseline runs/exp``.
"""

import json
from pathlib import Path

import optax
from flax import nnx
from tqdm import tqdm

from cli_args import parse_training_args, apply_args_to_config
from config import get_default_config, validate_config
from dataset import create_dataset
from network import Orion
from train import (
    create_optimizer,
    create_metrics, train_step, val_step, initialize_metrics_history,
)
from checkpoint import setup_checkpointing
from metrics_utils import format_metrics_for_history
from utils import plot_images, plot_training_curves


def create_validation_dataset(config):
    """Create the validation dataset (no augmentation).

    If val_data_dir differs from data_dir, validation images are loaded from
    the separate directory. Otherwise an image-level split of the training
    data is used (see dataset.load_images_to_memory).
    """
    val_config = config.copy()
    if config['val_data_dir'] != config['data_dir']:
        val_config['data_dir'] = config['val_data_dir']
        val_config['split'] = 'all'
        val_config['val_split'] = 0.0
    else:
        val_config['split'] = 'val'
    val_config['augmentation_prob'] = 0.0
    return create_dataset(val_config)


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _save_history(run_dir: Path, metrics_history: dict, config: dict) -> None:
    """Persist metrics history + a metadata header to <run_dir>/history.json."""
    payload = {
        'run_name': config.get('run_name'),
        'loss_weights': config.get('loss_weights'),
        'lr': config.get('lr'),
        'epochs': config.get('epochs'),
        'steps_per_epoch': config.get('steps_per_epoch'),
        'batch_size': config.get('batch_size'),
        'warmup_epochs': config.get('warmup_epochs'),
        'bottleneck_depth': config.get('bottleneck_depth'),
        'naf_expansion': config.get('naf_expansion'),
        'metrics': metrics_history,
    }
    _save_json(payload, run_dir / 'history.json')


def train_epoch(model, optimizer, train_metrics, train_dataset, config,
                step_fn, epoch, global_step, lr_schedule, metrics_history):
    """Train for one epoch. Returns the updated global_step."""
    pbar = tqdm(total=config['steps_per_epoch'],
                desc=f'Epoch {epoch+1}/{config["epochs"]} [TRAIN]')
    eval_every = config.get('eval_every', 2)

    for step, batch in enumerate(train_dataset):
        step_fn(batch)
        global_step += 1

        if step > 0 and (step % eval_every == 0):
            computed = train_metrics.compute()
            for name, value in format_metrics_for_history(computed, 'train_').items():
                metrics_history.setdefault(name, []).append(value)
            metrics_history['learning_rate'].append(float(lr_schedule(global_step)))
            train_metrics.reset()

        if step % 10 == 0:
            pbar.set_postfix({'lr': f"{lr_schedule(global_step):.2e}"})
        pbar.update(1)

        if step >= config['steps_per_epoch'] - 1:
            break

    pbar.close()
    return global_step


def validate_epoch(model, val_metrics, val_dataset, config, step_fn, epoch):
    """Run validation for one epoch. Returns computed validation metrics."""
    print("Running validation...")
    val_pbar = tqdm(total=config['val_steps'],
                    desc=f'Epoch {epoch+1}/{config["epochs"]} [VAL]')

    for v_step, batch in enumerate(val_dataset):
        step_fn(batch)
        val_pbar.update(1)
        if v_step >= config['val_steps'] - 1:
            break

    val_pbar.close()
    computed = val_metrics.compute()
    val_metrics.reset()
    return computed


def main():
    """Main training function."""
    args = parse_training_args()
    config = get_default_config()
    config = apply_args_to_config(args, config)
    validate_config(config)

    run_dir = Path(config['checkpoint_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_json(config, run_dir / 'config.json')
    show_plots = not config.get('no_show', False)

    print(f"Run: {config.get('run_name')} -> {run_dir}")

    print("Initializing datasets...")
    train_dataset = create_dataset(config)
    val_dataset = create_validation_dataset(config)

    print("Initializing model...")
    model = Orion(
        in_channels=config['in_channels'],
        bottleneck_depth=config['bottleneck_depth'],
        rngs=nnx.Rngs(0),
        vit_mlp_dropout_rate=config.get('vit_mlp_dropout_rate', 0.1),
        attn_dropout_rate=config.get('attn_dropout_rate', 0.0),
        stochastic_depth_rate=config.get('stochastic_depth_rate', 0.05),
        conv_dropout_rate=config.get('conv_dropout_rate', 0.05),
        naf_expansion=config.get('naf_expansion', 2),
        input_size=config.get('patch_size', 256),
    )

    print("Initializing optimizer...")
    warmup_steps = config['warmup_epochs'] * config['steps_per_epoch']
    total_steps = config['epochs'] * config['steps_per_epoch']
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=config['starting_lr'],
        peak_value=config['lr'],
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=config['starting_lr'],
    )
    optimizer = create_optimizer(model, config, lr_schedule)

    # Loss + metrics (Charbonnier + SSIM)
    loss_weights = config['loss_weights']
    component_names = list(loss_weights.keys())
    train_metrics = create_metrics(component_names)
    val_metrics = create_metrics(component_names)
    metrics_history = initialize_metrics_history(component_names)
    print(f"Loss components: {component_names} weights: {loss_weights}")

    ckpt_manager, start_epoch, best_val_loss, global_step = setup_checkpointing(
        config, model, optimizer
    )

    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch + 1}")
    print(f"{'='*60}\n")

    # Warm up / compile the jitted step functions once, outside the timed loop.
    print("Warming up (compiling step functions)...")

    def train_fn(b):
        train_step(model, optimizer, train_metrics, b, loss_weights)

    def val_fn(b):
        val_step(model, val_metrics, b, loss_weights)

    train_fn(next(iter(train_dataset)))
    val_fn(next(iter(val_dataset)))
    train_metrics.reset()
    val_metrics.reset()

    for epoch in range(start_epoch, config['epochs']):
        global_step = train_epoch(
            model, optimizer, train_metrics, train_dataset, config,
            train_fn, epoch, global_step, lr_schedule, metrics_history,
        )

        val_computed = validate_epoch(
            model, val_metrics, val_dataset, config, val_fn, epoch,
        )

        current_val_loss = float(val_computed['loss'])
        for name, value in format_metrics_for_history(val_computed, 'val_').items():
            metrics_history.setdefault(name, []).append(value)
        metrics_history['epochs'].append(epoch)

        print(f"Epoch {epoch+1} - Val Loss: {current_val_loss:.6e}")

        # Persist history every epoch so a killed run still has data.
        _save_history(run_dir, metrics_history, config)

        # Save periodic checkpoint
        if (epoch + 1) % config['epoch_checkpoints'] == 0:
            print("Saving checkpoint...")
            ckpt_manager.save_checkpoint(model, optimizer, epoch, best_val_loss, global_step)
            print("Checkpoint saved")

        # Save best model
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            print(f"New best validation loss: {best_val_loss:.6e}")
            ckpt_manager.save_best_model(model, optimizer, epoch, best_val_loss, global_step)

        # Visualization (training curves)
        if epoch % config['visualize_every'] == 0:
            plot_training_curves(metrics_history, show=show_plots,
                                 save_path=str(run_dir / 'curves.png'))

        # Plot sample predictions
        if epoch % config['plot_every'] == 0:
            batch = next(iter(val_dataset))
            orig, starless = batch
            prediction = model(orig[0][None, :], run=False)[0]
            plot_images(orig[0], starless[0], prediction, epoch,
                        show=show_plots,
                        save_path=str(run_dir / f'sample_epoch_{epoch+1:03d}.png'))

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6e}")
    print(f"Outputs saved in: {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
