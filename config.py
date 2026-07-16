"""Configuration management for Orion training."""

from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """Return the default training configuration."""
    return {
        # Data parameters
        "data_dir": "./data",
        "val_data_dir": "./data",  # Separate validation data dir; defaults to data_dir
        "patch_size": 256,
        "batch_size": 16,
        "steps_per_epoch": 250,
        "val_steps": 50,
        "val_split": 0.1,  # Fraction of images held out for validation (when val_data_dir == data_dir)
        "augmentation_prob": 0.5,
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
        "lr": 1e-4,
        "starting_lr": 1e-6,
        "warmup_epochs": 1,
        "momentum": 0.9,

        # Loss configuration (Charbonnier + SSIM)
        "loss_weights": {"charbonnier": 1.0, "ssim": 1.0},

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

    val_split = config.get('val_split', 0.0)
    if not (0.0 <= val_split < 1.0):
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")

    print("✓ Configuration validated successfully")
