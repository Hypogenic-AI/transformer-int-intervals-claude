"""Generate all visualizations from experiment results."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "../results"
FIGURES_DIR = "../results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 150,
})


def load_results():
    with open(os.path.join(RESULTS_DIR, "experiment_results.json")) as f:
        return json.load(f)


def plot_behavioral(results):
    """Bar chart of behavioral accuracy across tasks."""
    beh = results["behavioral"]
    tasks = ["comparison", "interval", "ring"]
    accs = [beh[t] for t in tasks]
    labels = ["Number\nComparison", "Interval\nMembership", "Ring\nCounting"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accs, color=['#4C72B0', '#DD8452', '#55A868'], width=0.6, edgecolor='black')
    ax.set_ylabel("Accuracy")
    ax.set_title("GPT-2 Small: Behavioral Accuracy on Numerical Tasks")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (binary)')
    ax.axhline(1/26, color='gray', linestyle=':', alpha=0.7, label='Chance (26-class)')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', fontweight='bold')

    ax.legend(loc='upper right')
    plt.savefig(os.path.join(FIGURES_DIR, "behavioral_accuracy.png"))
    plt.close()
    print("Saved behavioral_accuracy.png")


def plot_probing_comparison(results):
    """Line plot of probe accuracy by layer for comparison, interval, and ring tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Comparison probing
    ax = axes[0]
    layers = sorted([int(k) for k in results["probe_comparison"]["real"].keys()])
    real_accs = [results["probe_comparison"]["real"][str(l)] for l in layers]
    ctrl_accs = [results["probe_comparison"]["control"][str(l)] for l in layers]
    ax.plot(layers, real_accs, 'o-', color='#4C72B0', label='Real labels', linewidth=2)
    ax.plot(layers, ctrl_accs, 's--', color='gray', alpha=0.6, label='Shuffled (control)')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Comparison (A > B?)")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.5)

    # Interval probing
    ax = axes[1]
    layers = sorted([int(k) for k in results["probe_interval"]["real"].keys()])
    real_accs = [results["probe_interval"]["real"][str(l)] for l in layers]
    ctrl_accs = [results["probe_interval"]["control"][str(l)] for l in layers]
    ax.plot(layers, real_accs, 'o-', color='#DD8452', label='Real labels', linewidth=2)
    ax.plot(layers, ctrl_accs, 's--', color='gray', alpha=0.6, label='Shuffled (control)')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Interval Membership (X ∈ [A,B]?)")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.5)

    # Ring probing
    ax = axes[2]
    layers = sorted([int(k) for k in results["probe_ring"].keys()])
    ring_accs = [results["probe_ring"][str(l)] for l in layers]
    ax.plot(layers, ring_accs, 'o-', color='#55A868', label='Real labels', linewidth=2)
    ax.axhline(1/26, color='red', linestyle=':', alpha=0.5, label=f'Chance ({1/26:.1%})')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Ring Counting (26-class)")
    ax.legend()
    ax.set_ylim(0, max(ring_accs) * 1.3)

    plt.suptitle("Linear Probe Accuracy by Layer (GPT-2 Small)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "probing_by_layer.png"))
    plt.close()
    print("Saved probing_by_layer.png")


def plot_regression_probing(results):
    """Plot R² for numerical value decoding and offset decoding by layer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # X value decoding
    ax = axes[0]
    layers = sorted([int(k) for k in results["probe_x_value"].keys()])
    r2s = [results["probe_x_value"][str(l)] for l in layers]
    ax.plot(layers, r2s, 'o-', color='#4C72B0', linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Decoding X Value from Interval Task")
    ax.axhline(0, color='red', linestyle=':', alpha=0.5)
    ax.set_ylim(min(min(r2s) - 0.05, -0.1), max(max(r2s) + 0.05, 0.5))

    # Offset decoding
    ax = axes[1]
    layers = sorted([int(k) for k in results["probe_ring_offset"].keys()])
    r2s = [results["probe_ring_offset"][str(l)] for l in layers]
    ax.plot(layers, r2s, 'o-', color='#55A868', linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Decoding Offset N from Ring Task")
    ax.axhline(0, color='red', linestyle=':', alpha=0.5)
    ax.set_ylim(min(min(r2s) - 0.05, -0.1), max(max(r2s) + 0.05, 0.5))

    plt.suptitle("Regression Probe R² by Layer (GPT-2 Small)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "regression_probing.png"))
    plt.close()
    print("Saved regression_probing.png")


def plot_width_analysis(results):
    """Plot interval membership probe accuracy by width and layer."""
    width_data = results["probe_interval_by_width"]
    widths = sorted([int(w) for w in width_data.keys()])
    # Get layers from first width entry
    first_width = str(widths[0])
    layers = sorted([int(l) for l in width_data[first_width].keys()])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(widths)))

    for w, color in zip(widths, colors):
        accs = [width_data[str(w)][str(l)] for l in layers]
        ax.plot(layers, accs, 'o-', color=color, label=f'Width={w}', linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Interval Membership Probe Accuracy by Interval Width")
    ax.legend(title="Interval Width")
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(FIGURES_DIR, "width_analysis.png"))
    plt.close()
    print("Saved width_analysis.png")


def plot_patching(results):
    """Plot activation patching effects by layer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title, color in [
        (axes[0], "patching_comparison", "Comparison Task", '#4C72B0'),
        (axes[1], "patching_interval", "Interval Membership Task", '#DD8452'),
    ]:
        layers = sorted([int(k) for k in results[key].keys()])
        effects = [results[key][str(l)] for l in layers]
        ax.bar(layers, effects, color=color, edgecolor='black', alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Patching Effect\n(logit difference shift)")
        ax.set_title(f"Activation Patching: {title}")
        ax.axhline(0, color='black', linewidth=0.5)

    plt.suptitle("Causal Importance by Layer (Activation Patching)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "activation_patching.png"))
    plt.close()
    print("Saved activation_patching.png")


def plot_combined_probing(results):
    """Single plot comparing probing across all three tasks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Comparison
    layers_c = sorted([int(k) for k in results["probe_comparison"]["real"].keys()])
    real_c = [results["probe_comparison"]["real"][str(l)] for l in layers_c]
    ax.plot(layers_c, real_c, 'o-', color='#4C72B0', label='Comparison', linewidth=2)

    # Interval
    layers_i = sorted([int(k) for k in results["probe_interval"]["real"].keys()])
    real_i = [results["probe_interval"]["real"][str(l)] for l in layers_i]
    ax.plot(layers_i, real_i, 's-', color='#DD8452', label='Interval Membership', linewidth=2)

    # Ring (rescaled to [0,1] from 26-class)
    layers_r = sorted([int(k) for k in results["probe_ring"].keys()])
    ring_accs = [results["probe_ring"][str(l)] for l in layers_r]
    # Normalize: (acc - chance) / (1 - chance) to make comparable
    chance_26 = 1/26
    ring_normalized = [(a - chance_26) / (1 - chance_26) for a in ring_accs]
    # Shift to 0.5+ scale for visual comparison
    ring_display = [0.5 + 0.5 * r for r in ring_normalized]
    ax.plot(layers_r, ring_display, '^-', color='#55A868', label='Ring Counting (normalized)', linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy (or normalized equivalent)")
    ax.set_title("Information Emergence Across Layers (GPT-2 Small)")
    ax.legend()
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='Chance (binary)')
    plt.savefig(os.path.join(FIGURES_DIR, "combined_probing.png"))
    plt.close()
    print("Saved combined_probing.png")


def generate_all_plots():
    results = load_results()
    plot_behavioral(results)
    plot_probing_comparison(results)
    plot_regression_probing(results)
    plot_width_analysis(results)
    plot_patching(results)
    plot_combined_probing(results)
    print(f"\nAll plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
