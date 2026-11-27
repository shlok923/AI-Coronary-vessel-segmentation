"""
Plot Training Loss vs. Epoch for all three models (M1, M2, M2+I2).

Usage:
    python plot_training_loss.py
    python plot_training_loss.py --smooth 0.6
    python plot_training_loss.py --runs_dir results/runs --out_dir results/figures
"""
import os
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Model definitions with display names and distinctive publication-quality colors
MODEL_CONFIGS = [
    {
        'id': 'baseline_unet',
        'name': 'Baseline U-Net (M1)',
        'color': '#1f77b4',       # Blue
        'linestyle': '-',
        'marker': 'o',
    },
    {
        'id': 'hybrid_unet',
        'name': 'Hybrid CNN-Mamba (M2)',
        'color': '#d62728',       # Red/Crimson
        'linestyle': '--',
        'marker': 's',
    },
    {
        'id': 'hybrid_unet_i2',
        'name': 'Hybrid Mamba + I2 (Proposed)',
        'color': '#2ca02c',       # Emerald Green
        'linestyle': '-.',
        'marker': '^',
    },
]


def find_metrics_file(model_id, runs_dir='results/runs'):
    """Locate metrics.csv for a model across local and drive paths."""
    candidates = [
        os.path.join(runs_dir, model_id, 'metrics.csv'),
        os.path.join('results/runs', model_id, 'metrics.csv'),
        os.path.join('/content/drive/MyDrive/AI-Coronary-vessel-segmentation/results/runs', model_id, 'metrics.csv'),
        os.path.join('/content/drive/MyDrive/coronary-vessel-segmentation/results/runs', model_id, 'metrics.csv'),
        os.path.join('/content/AI-Coronary-vessel-segmentation/results/runs', model_id, 'metrics.csv'),
        os.path.join('/content/coronary-vessel-segmentation/results/runs', model_id, 'metrics.csv'),
    ]
    # Check Downloads folder fallback if running locally
    downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    if os.path.isdir(downloads_dir):
        for p in glob.glob(os.path.join(downloads_dir, '*results*', 'results', 'runs', model_id, 'metrics.csv')):
            candidates.append(p)

    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def smooth_curve(values, weight=0.6):
    """Exponential Moving Average (EMA) smoothing for clean training curves."""
    if weight <= 0 or len(values) == 0:
        return values
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_training_loss_only(models_data, out_path, smooth_weight=0.0):
    """Generate focused Training Loss vs Epoch plot comparing all 3 models."""
    plt.figure(figsize=(9, 6), dpi=300)
    ax = plt.subplot(1, 1, 1)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.35, color='gray')

    max_epoch = 1
    for m in models_data:
        df = m['df']
        epochs = df['epoch'].values
        loss = df['train_loss'].values
        max_epoch = max(max_epoch, int(epochs.max()))

        label_text = f"{m['name']} (min: {loss.min():.4f})"

        if smooth_weight > 0 and len(loss) > 5:
            smoothed = smooth_curve(loss, weight=smooth_weight)
            # Plot raw with transparency and smoothed as solid line
            ax.plot(epochs, loss, color=m['color'], alpha=0.25, linewidth=1.0)
            ax.plot(epochs, smoothed, color=m['color'], linestyle=m['linestyle'],
                    linewidth=2.2, label=label_text)
        else:
            # Mark every 10th epoch with symbol
            markevery = max(1, len(epochs) // 10)
            ax.plot(epochs, loss, color=m['color'], linestyle=m['linestyle'],
                    linewidth=2.0, marker=m['marker'], markevery=markevery,
                    markersize=5, label=label_text)

        # Highlight minimum loss point
        min_idx = np.argmin(loss)
        ax.scatter(epochs[min_idx], loss[min_idx], color=m['color'], s=60, zorder=5)

    ax.set_title("Training Loss vs. Epoch across Models (DCA1 Dataset)",
                 fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel("Epoch", fontsize=12, fontweight='medium', labelpad=8)
    ax.set_ylabel("Training Loss", fontsize=12, fontweight='medium', labelpad=8)
    ax.set_xlim(1, max_epoch)
    ax.set_ylim(bottom=0)

    # Position legend cleanly
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10,
              loc='upper right', edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Training Loss graph to: {out_path}")


def plot_train_val_comparison(models_data, out_path):
    """Plot side-by-side Train Loss and Val Loss comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # 1. Train Loss
    ax1 = axes[0]
    ax1.set_title("Training Loss vs. Epoch", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Training Loss", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Val Loss
    ax2 = axes[1]
    ax2.set_title("Validation Loss vs. Epoch", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.35)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for m in models_data:
        df = m['df']
        epochs = df['epoch'].values
        train_loss = df['train_loss'].values
        val_loss = df['val_loss'].values

        markevery = max(1, len(epochs) // 10)
        ax1.plot(epochs, train_loss, color=m['color'], linestyle=m['linestyle'],
                 linewidth=1.8, marker=m['marker'], markevery=markevery,
                 markersize=4, label=f"{m['name']} (min: {train_loss.min():.4f})")
        ax2.plot(epochs, val_loss, color=m['color'], linestyle=m['linestyle'],
                 linewidth=1.8, marker=m['marker'], markevery=markevery,
                 markersize=4, label=f"{m['name']} (min: {val_loss.min():.4f})")

    ax1.legend(frameon=True, fontsize=9, loc='upper right')
    ax2.legend(frameon=True, fontsize=9, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Train vs. Val Loss comparison to: {out_path}")


def plot_full_dashboard(models_data, out_path):
    """Plot 4-panel dashboard: Train Loss, Val Loss, Val Dice, Val clDice."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    panels = [
        ('train_loss', 'Training Loss', axes[0, 0], False),
        ('val_loss',   'Validation Loss', axes[0, 1], False),
        ('val_dice',   'Validation Dice (Area Overlap)', axes[1, 0], True),
        ('val_cldice', 'Validation clDice (Centerline Topology)', axes[1, 1], True),
    ]

    for col, title, ax, higher_is_better in panels:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for m in models_data:
            df = m['df']
            if col not in df.columns:
                continue
            epochs = df['epoch'].values
            vals = df[col].values
            markevery = max(1, len(epochs) // 10)
            best_val = vals.max() if higher_is_better else vals.min()
            tag = "max" if higher_is_better else "min"
            ax.plot(epochs, vals, color=m['color'], linestyle=m['linestyle'],
                    linewidth=1.8, marker=m['marker'], markevery=markevery,
                    markersize=4, label=f"{m['name']} ({tag}: {best_val:.4f})")

        ax.legend(frameon=True, fontsize=8, loc='best')

    plt.suptitle("Training Progress Dashboard Across Models (DCA1 Dataset)",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved 4-panel training dashboard to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Training Loss vs. Epoch for M1, M2, and M2+I2.")
    parser.add_argument('--runs_dir', type=str, default='results/runs',
                        help="Path to results/runs directory containing model subfolders.")
    parser.add_argument('--out_dir', type=str, default='results/figures',
                        help="Output directory for generated plots.")
    parser.add_argument('--smooth', type=float, default=0.0,
                        help="Optional EMA smoothing factor between 0.0 and 0.9 (e.g. 0.6).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    models_data = []
    print(f"[Plot] Searching for model run logs under '{args.runs_dir}'...")

    for m in MODEL_CONFIGS:
        csv_path = find_metrics_file(m['id'], args.runs_dir)
        if csv_path:
            df = pd.read_csv(csv_path)
            print(f"  ✓ Found {m['name']} ({len(df)} epochs): {csv_path}")
            models_data.append({
                'id': m['id'],
                'name': m['name'],
                'color': m['color'],
                'linestyle': m['linestyle'],
                'marker': m['marker'],
                'df': df,
                'path': csv_path
            })
        else:
            print(f"  ✗ Missing log for {m['name']} ({m['id']}/metrics.csv)")

    if not models_data:
        print("\n[Error] No metrics.csv files found. Ensure 'results/runs' exists or check paths.")
        return

    # 1. Primary requested plot: Training Loss vs. Epoch
    loss_out = os.path.join(args.out_dir, 'training_loss_vs_epoch.png')
    plot_training_loss_only(models_data, loss_out, smooth_weight=args.smooth)

    # 2. Train vs. Val Loss comparison
    val_comp_out = os.path.join(args.out_dir, 'loss_train_val_comparison.png')
    plot_train_val_comparison(models_data, val_comp_out)

    # 3. Complete 4-panel training dashboard
    dashboard_out = os.path.join(args.out_dir, 'training_dashboard.png')
    plot_full_dashboard(models_data, dashboard_out)

    print("\n" + "=" * 60)
    print("Generated Graphs:")
    print(f"  1. Training Loss:        {loss_out}")
    print(f"  2. Train & Val Loss:     {val_comp_out}")
    print(f"  3. Complete Dashboard:   {dashboard_out}")
    print("=" * 60)


if __name__ == '__main__':
    main()
