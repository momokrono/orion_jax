"""Configuration management for Orion training."""

from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """Return the default training configuration."""
    return {
        # Data parameters
        "data_dir": "./data",
        "val_data_dir": "./data",  # Separate validation data dir; defaults to data_dir
        "patch_size": 256,
        "batch_size": 8,  # Fits a 12 GB GPU in bf16; raise on larger cards
        "steps_per_epoch": 250,
        "val_steps": 50,
        "val_split": 0.1,  # Fraction of images held out for validation (when val_data_dir == data_dir)
        "augmentation_prob": 1.0,  # Per-patch probability of photometric jitter; defaults to 1.0 (always on). D4 orientation aug is always on for train regardless.
        # Photometric jitter strengths (gamma + gain applied GLOBALLY, identically
        # to R/G/B). Per-channel independence was tried and regressed held-out
        # quality (it breaks the spectral signature the model uses to tell
        # emission nebulosity apart from stars); see AstroDataset._photometric.
        "photo_gamma_strength": 0.7,  # half-width of log-uniform gamma: 2^U(-g, g) ~= 0.62..1.62
        "photo_gain_strength": 0.2,   # half-width of additive gain: 1 + U(-a, a) ~= 0.8..1.2
        "prefetch": 2,

        # Model parameters
        "in_channels": 3,
        "bottleneck_depth": 2,
        "naf_expansion": 2,

        # Regularization
        "vit_mlp_dropout_rate": 0.1,       # Dropout in ViT MLP (0.0 to disable)
        "attn_dropout_rate": 0.0,          # Dropout in MHSA (0.0 to disable)
        "stochastic_depth_rate": 0.05,     # Drop-path across bottleneck layers (0.0 to disable)
        "conv_dropout_rate": 0.05,         # Dropout in conv residual blocks (0.0 to disable)

        # Training parameters
        "epochs": 10,
        "lr": 5e-4,
        "starting_lr": 1e-6,
        "warmup_epochs": 1,
        "momentum": 0.9,
        # Continuation (--continue-epochs): peak LR for the additional epochs,
        # expressed as a fraction of `lr`. The cosine schedule decays from
        # `lr * continue_lr_scale` to `starting_lr` over the extra epochs.
        "continue_lr_scale": 0.3,

        # Loss configuration (weighted sum; see loss_functions.LOSS_FUNCTIONS).
        # log_charbonnier + gradient + frequency specifically strengthen the
        # signal on faint/low-contrast stars that plain pixel losses ignore.
        "loss_weights": {
            "charbonnier": 1.0,       # robust pixel reconstruction (linear)
            "log_charbonnier": 1.0,   # log-space reconstruction — boosts faint stars
            "ssim": 1.0,             # structural similarity
            "gradient": 1.0,         # edge/structure preservation (star spikes)
            "frequency": 0.3,        # residual point-source (high-freq) penalty
        },

        # Precision: 'bf16' (mixed-precision compute, faster, less GPU memory)
        # or 'fp32'. Master weights / optimizer state stay fp32 either way.
        "precision": "bf16",

        # Optimization
        "grad_clip": 1.0,          # Global gradient clipping norm (None to disable)

        # Logging and checkpointing
        "eval_every": 2,           # Evaluate metrics every N steps
        "visualize_every": 5,      # Show plots every N epochs
        "plot_every": 5,           # Show sample predictions every N epochs
        "epoch_checkpoints": 5,    # Save checkpoint every N epochs
        "runs_root": "./runs",     # Parent dir for per-run outputs
        "run_name": "default",     # Run label; checkpoint_dir = runs_root/run_name
        "checkpoint_dir": "./runs/default",  # Resolved per-run output directory
        "resume_from_checkpoint": True,  # Auto-resume if checkpoint exists
    }


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration; raise ValueError if invalid."""
    required_keys = [
        "data_dir", "patch_size", "batch_size", "steps_per_epoch",
        "epochs", "lr", "starting_lr", "warmup_epochs", "loss_weights",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config["lr"] <= 0:
        raise ValueError(f"Learning rate must be positive, got {config['lr']}")

    if config["starting_lr"] <= 0:
        raise ValueError(f"Starting LR must be positive, got {config['starting_lr']}")

    if config["starting_lr"] > config["lr"]:
        raise ValueError(f"Starting LR ({config['starting_lr']}) should be <= peak LR ({config['lr']})")

    if config["batch_size"] <= 0:
        raise ValueError(f"Batch size must be positive, got {config['batch_size']}")

    if config["epochs"] <= 0:
        raise ValueError(f"Epochs must be positive, got {config['epochs']}")

    if config["warmup_epochs"] > config["epochs"]:
        raise ValueError(f"Warmup epochs ({config['warmup_epochs']}) > total epochs ({config['epochs']})")

    if not config.get("loss_weights"):
        raise ValueError("loss_weights must be a non-empty dict (e.g. {'charbonnier': 1.0, 'ssim': 1.0})")

    from loss_functions import LOSS_FUNCTIONS
    unknown = [k for k in config["loss_weights"] if k not in LOSS_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Unknown loss_weights keys {unknown}. Valid: {sorted(LOSS_FUNCTIONS)}"
        )

    val_split = config.get('val_split', 0.0)
    if not (0.0 <= val_split < 1.0):
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")

    print("✓ Configuration validated successfully")
