"""Compare multiple Orion training runs side-by-side.

Each run directory is expected to contain a ``history.json`` written by
``main.py``. Example:

    python compare_runs.py runs/baseline runs/structure_focused
    python compare_runs.py --runs-root runs --pattern '*'   # all runs
    python compare_runs.py runs/baseline --out cmp.png       # save, no display

Plots overlaid validation loss, validation loss components, the learning-rate
schedule, and final train/val loss summary table.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


def load_run(run_dir: Path) -> Optional[dict]:
    """Load a run's history.json; return None if missing/malformed."""
    path = run_dir / 'history.json'
    if not path.exists():
        print(f"  [skip] {run_dir} : no history.json", file=sys.stderr)
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as e:
        print(f"  [skip] {run_dir} : bad json ({e})", file=sys.stderr)
        return None
    payload['_dir'] = str(run_dir)
    return payload


def discover_runs(args) -> List[Path]:
    if args.runs:
        return [Path(r) for r in args.runs]
    root = Path(args.runs_root)
    if not root.exists():
        sys.exit(f"runs root not found: {root}")
    return sorted(p for p in root.glob(args.pattern) if (p / 'history.json').exists())


def short_label(run: dict, fallback: str) -> str:
    name = run.get('run_name') or fallback
    preset = run.get('loss_preset')
    return f"{name}" + (f" [{preset}]" if preset else "")


def plot_total_loss(runs: List[dict], ax) -> None:
    for run in runs:
        metrics = run['metrics']
        val = metrics.get('val_loss', [])
        if not val:
            continue
        epochs = metrics.get('epochs', list(range(1, len(val) + 1)))
        epochs = [e + 1 for e in epochs]  # 1-indexed for display
        label = short_label(run, Path(run['_dir']).name)
        ax.plot(epochs, val, marker='o', linewidth=2, label=label)
    ax.set_title('Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val loss (log)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_train_loss(runs: List[dict], ax) -> None:
    for run in runs:
        metrics = run['metrics']
        train = metrics.get('train_loss', [])
        if not train:
            continue
        label = short_label(run, Path(run['_dir']).name)
        ax.semilogy(train, label=label, alpha=0.8)
    ax.set_title('Train loss (per eval window)')
    ax.set_xlabel('Eval window')
    ax.set_ylabel('Train loss (log)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_components(runs: List[dict], axes) -> None:
    """Plot each val_<component> as its own subplot, runs overlaid."""
    component_names = set()
    for run in runs:
        for k in run['metrics']:
            if k.startswith('val_') and k != 'val_loss':
                component_names.add(k[len('val_'):])
    component_names = sorted(component_names)

    for ax, name in zip(axes, component_names):
        for run in runs:
            series = run['metrics'].get(f'val_{name}', [])
            if not series:
                continue
            epochs = run['metrics'].get('epochs', list(range(1, len(series) + 1)))
            epochs = [e + 1 for e in epochs]
            label = short_label(run, Path(run['_dir']).name)
            ax.plot(epochs, series, marker='o', linewidth=2, label=label)
        ax.set_title(f'Val {name}')
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # hide unused axes
    for ax in axes[len(component_names):]:
        ax.axis('off')


def plot_lr(runs: List[dict], ax) -> None:
    for run in runs:
        lr = run['metrics'].get('learning_rate', [])
        if not lr:
            continue
        label = short_label(run, Path(run['_dir']).name)
        ax.plot(lr, label=label, alpha=0.8)
    ax.set_title('Learning rate schedule')
    ax.set_xlabel('Eval window')
    ax.set_ylabel('LR (log)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()


def print_summary(runs: List[dict]) -> None:
    print("\nSummary (final values)")
    print("=" * 78)
    fmt = "{:<28s} {:>10s} {:>12s} {:>10s} {:>10s}\n"
    print(fmt.format("Run", "best_val", "final_val", "epochs", "lr_peak"))
    print("-" * 78)
    for run in runs:
        name = short_label(run, Path(run['_dir']).name)
        metrics = run['metrics']
        val = metrics.get('val_loss', [])
        if not val:
            continue
        best = min(val)
        final = val[-1]
        lr_peak = max(metrics.get('learning_rate', [0.0]))
        n_epochs = len(val)
        print("{:<28s} {:>10.4e} {:>12.4e} {:>10d} {:>10.2e}".format(
            name[:28], best, final, n_epochs, lr_peak))
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*', help='Run directories to compare.')
    p.add_argument('--runs-root', default='./runs',
                   help='Root dir to glob when no run dirs given (default: ./runs).')
    p.add_argument('--pattern', default='*',
                   help='Glob pattern under --runs-root (default: *).')
    p.add_argument('--out', default=None,
                   help='Save the comparison figure to this path.')
    p.add_argument('--no-show', action='store_true', help='Skip plt.show().')
    args = p.parse_args()

    run_dirs = discover_runs(args)
    if not run_dirs:
        sys.exit("no runs found")

    print(f"Loading {len(run_dirs)} run(s):")
    runs = []
    for d in run_dirs:
        run = load_run(d)
        if run is not None:
            runs.append(run)
            print(f"  - {d} ({short_label(run, d.name)})")
    if not runs:
        sys.exit("no valid runs loaded")

    # Layout: total loss, train loss, lr, then per-component subplots.
    component_count = 0
    for run in runs:
        component_count = max(component_count, sum(
            1 for k in run['metrics']
            if k.startswith('val_') and k != 'val_loss'))
    n_plots = 3 + component_count
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    plot_total_loss(runs, axes[0])
    plot_train_loss(runs, axes[1])
    plot_lr(runs, axes[2])
    plot_components(runs, axes[3:])

    for ax in axes[3 + component_count:]:
        ax.axis('off')

    plt.tight_layout()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=120)
        print(f"\nSaved comparison plot -> {args.out}")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    print_summary(runs)


if __name__ == "__main__":
    main()
