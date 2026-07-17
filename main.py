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

import jax.numpy as jnp
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
from checkpoint import setup_checkpointing, CheckpointManager
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


def _build_continuation_schedule(config: dict) -> optax.Schedule:
    """Build the LR schedule for ``--continue-epochs``.

    Peeks at the existing ``latest`` checkpoint to find the resume point
    (``global_step`` / ``epoch``), then constructs a cosine schedule that
    decays from a re-heated peak (``lr * continue_lr_scale``) down to
    ``starting_lr`` over exactly ``continue_epochs * steps_per_epoch`` steps.
    The schedule is offset by ``resume_step`` so that when the optimizer's
    restored step counter ticks past the resume point, the cosine starts at
    its peak.

    Side effects:
        - Updates ``config['epochs']`` to ``resume_epoch + 1 + continue_epochs``
          so the training loop runs the extra epochs.
        - Sets ``config['resume_from_checkpoint'] = True`` (required).

    Returns:
        The optax.Schedule to feed into ``create_optimizer``.
    """
    ckpt_dir = config['checkpoint_dir']
    continue_epochs = int(config['continue_epochs'])
    metadata = CheckpointManager(ckpt_dir).peek_metadata('latest')
    if metadata is None:
        raise SystemExit(
            f"--continue-epochs: no 'latest' checkpoint found in {ckpt_dir}. "
            f"Run a fresh training first, or pass --epochs instead."
        )

    resume_step = metadata['global_step']
    resume_epoch = metadata['epoch']  # last completed epoch (0-indexed)
    extra_steps = continue_epochs * config['steps_per_epoch']
    reheated_peak = config['lr'] * config.get('continue_lr_scale', 0.3)

    # Offset cosine: at step `s`, the cosine has progressed by max(s - resume_step, 0).
    # So when the restored optimizer ticks past resume_step, the new cosine begins
    # cleanly at its peak. warmup_steps=0 (no second warmup), init=peak so the
    # schedule starts at full reheated_peak immediately.
    inner = optax.warmup_cosine_decay_schedule(
        init_value=reheated_peak,
        peak_value=reheated_peak,
        warmup_steps=0,
        decay_steps=extra_steps,
        end_value=config['starting_lr'],
    )

    def schedule(step):
        return inner(jnp.maximum(step - resume_step, 0))

    # Extend the loop so it runs `continue_epochs` more epochs past the resume point.
    config['epochs'] = resume_epoch + 1 + continue_epochs
    config['resume_from_checkpoint'] = True

    print(f"[continue] resuming at global_step={resume_step} (epoch {resume_epoch}); "
          f"adding {continue_epochs} epoch(s) = {extra_steps} steps; "
          f"new total epochs = {config['epochs']}; "
          f"continuation peak LR = {reheated_peak:.2e} "
          f"({config.get('continue_lr_scale', 0.3)} x lr)")
    return schedule


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

    # --continue-epochs: extend an existing run by N more epochs with a fresh
    # re-heated cosine schedule. Mutually exclusive with --epochs, and requires
    # resume to be enabled (the whole point is to pick up an existing checkpoint).
    if args.continue_epochs is not None:
        if args.continue_epochs <= 0:
            raise SystemExit(f"--continue-epochs must be positive, got {args.continue_epochs}")
        if args.epochs is not None:
            raise SystemExit("--continue-epochs is mutually exclusive with --epochs "
                             "(--continue-epochs extends an existing run; --epochs sets "
                             "the total for a fresh run).")
        if args.no_resume:
            raise SystemExit("--continue-epochs requires checkpoint resume; "
                             "drop --no-resume.")
        config['continue_epochs'] = int(args.continue_epochs)

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
    compute_dtype = jnp.bfloat16 if config.get('precision', 'fp32') == 'bf16' else jnp.float32
    print(f"Compute dtype: {compute_dtype}")
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
        input_size=config.get('patch_size', 256),
    )

    print("Initializing optimizer...")
    if 'continue_epochs' in config:
        lr_schedule = _build_continuation_schedule(config)
    else:
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
