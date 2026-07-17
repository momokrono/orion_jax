"""Command-line argument parsing for Orion training.

All argparse defaults are ``None`` so that :func:`config.get_default_config`
remains the single source of truth; only flags passed explicitly on the
command line override the config. Use ``--name`` to give a run a label, which
also namespaces its checkpoint/history directory.
"""

import argparse
import os


def parse_training_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for training.

    Pass ``argv`` (a list of strings) for testing; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Train the Orion star removal network."
    )

    # Run identity
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run label. Namespaces checkpoint/history under <runs-root>/<name>/. "
             "Defaults to 'default'.",
    )
    parser.add_argument(
        "--runs-root", type=str, default=None,
        help="Root directory holding per-run subdirectories (default: ./runs).",
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate after warmup.")
    parser.add_argument("--starting-lr", type=float, default=None,
                        help="Starting LR for warmup and final decay.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs.")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Warmup epochs before cosine decay.")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Training steps per epoch.")
    parser.add_argument("--val-steps", type=int, default=None,
                        help="Validation steps per epoch.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size.")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Random crop size.")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing training data.")
    parser.add_argument("--val-data-dir", type=str, default=None,
                        help="Validation data dir (defaults to --data-dir).")
    parser.add_argument("--augmentation-prob", type=float, default=None,
                        help="Probability of photometric jitter per patch (default 1.0).")
    parser.add_argument("--photo-gamma-strength", type=float, default=None,
                        help="Half-width of per-channel log-uniform gamma stretch "
                             "(default 0.7 -> 2^U(-0.7,0.7) per channel).")
    parser.add_argument("--photo-gain-strength", type=float, default=None,
                        help="Half-width of per-channel additive multiplicative gain "
                             "(default 0.2 -> U(0.8, 1.2) per channel).")

    # Model architecture
    parser.add_argument("--bottleneck-depth", type=int, default=None,
                        help="Number of HybridLayer blocks in the bottleneck.")
    parser.add_argument("--naf-expansion", type=int, default=None,
                        help="Channel expansion in NAFNet blocks (default 2).")
    parser.add_argument("--precision", type=str, default=None,
                        choices=["bf16", "fp32"],
                        help="Compute precision: 'bf16' (default) or 'fp32'.")

    # Logging and visualization
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Log training metrics every N steps.")
    parser.add_argument("--visualize-every", type=int, default=None,
                        help="Plot training curves every N epochs.")
    parser.add_argument("--plot-every", type=int, default=None,
                        help="Render sample predictions every N epochs.")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable interactive plt.show() (headless runs).")

    # Checkpointing
    parser.add_argument("--epoch-checkpoints", type=int, default=None,
                        help="Save checkpoint every N epochs.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable automatic checkpoint resume.")

    return parser.parse_args(argv)


def apply_args_to_config(args, config):
    """Apply non-None CLI args on top of a config dict (in place).

    Also resolves the per-run directory: ``<runs_root>/<name>/`` becomes
    ``config['checkpoint_dir']``, and ``config['run_name']`` / ``config['runs_root']``
    are stored for downstream tooling.
    """
    _INT_KEYS = [
        'epochs', 'warmup_epochs', 'steps_per_epoch', 'val_steps',
        'batch_size', 'patch_size', 'bottleneck_depth', 'naf_expansion',
        'eval_every', 'visualize_every', 'plot_every', 'epoch_checkpoints',
    ]
    _FLOAT_KEYS = ['lr', 'starting_lr', 'augmentation_prob',
                   'photo_gamma_strength', 'photo_gain_strength']
    _STR_KEYS = ['data_dir', 'precision']

    for k in _INT_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            config[k] = int(v)
    for k in _FLOAT_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            config[k] = float(v)
    for k in _STR_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            config[k] = v

    if args.data_dir is not None:
        config['data_dir'] = args.data_dir
    if args.val_data_dir is not None:
        config['val_data_dir'] = args.val_data_dir
    else:
        config.setdefault('val_data_dir', config['data_dir'])

    if args.no_resume:
        config['resume_from_checkpoint'] = False
    if args.no_show:
        config['no_show'] = True

    # Per-run directory layout: <runs_root>/<name>/
    runs_root = args.runs_root or config.get('runs_root', './runs')
    name = args.name or config.get('run_name', 'default')
    config['runs_root'] = runs_root
    config['run_name'] = name
    config['checkpoint_dir'] = os.path.join(runs_root, name)

    return config
