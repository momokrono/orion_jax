"""Utilities for metrics formatting."""


def format_metrics_for_history(computed_metrics: dict, prefix: str = '') -> dict:
    """Format computed metrics for history tracking.

    Args:
        computed_metrics: Dictionary of computed metric values.
        prefix: Prefix to add to metric names (e.g. 'train_' or 'val_').

    Returns:
        Dictionary with formatted metric names and float values.
    """
    return {
        f"{prefix}{name}": float(value)
        for name, value in computed_metrics.items()
    }
