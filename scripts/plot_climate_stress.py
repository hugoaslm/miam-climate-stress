"""
Plotting script for climate-modality stress test results.

Generates 4-5 publication-quality figures from the evaluation CSV.

Usage:
    python scripts/plot_climate_stress.py --results results/climate_stress_results.csv
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

METHOD_COLORS = {
    "miam": "#2ecc71",       # green
    "dropout": "#e74c3c",    # red
    "constant": "#3498db",   # blue
    "opm": "#9b59b6",        # purple
    "dirichlet": "#f39c12",  # orange
    "uniform": "#1abc9c",    # teal
}

METHOD_LABELS = {
    "miam": "MIAM (ours)",
    "dropout": "Modality Dropout",
    "constant": "Constant Masking",
    "opm": "OPM",
    "dirichlet": "Dirichlet",
    "uniform": "Uniform",
}


def load_results(csv_path: str) -> pd.DataFrame:
    """Load and validate results CSV."""
    df = pd.read_csv(csv_path)

    required_cols = {"method", "condition", "auroc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Add derived columns
    if "clean" in df["condition"].values:
        clean_scores = df[df["condition"] == "clean"].set_index("method")["auroc"]
        df["clean_auroc"] = df["method"].map(clean_scores)
        df["auroc_drop"] = df["clean_auroc"] - df["auroc"]
        df["robustness_ratio"] = df["auroc"] / df["clean_auroc"]

    return df


# ── Figure 1: Clean vs Stressed Performance ──────────────────────────────

def plot_clean_vs_stress(df: pd.DataFrame, output_path: str):
    """Bar chart comparing methods under key stress conditions."""
    key_conditions = [
        "clean",
        "climate_missing",
        "climate_noise_050",
        "climate_shift_p1",
        "satellite_missing",
    ]

    plot_df = df[df["condition"].isin(key_conditions)].copy()

    condition_labels = {
        "clean": "Clean\n(all modalities)",
        "climate_missing": "Climate\nmissing",
        "climate_noise_050": "Climate\nnoisy (σ=0.5)",
        "climate_shift_p1": "Climate\nshifted (+1σ)",
        "satellite_missing": "Satellite\nmissing",
    }
    plot_df["condition_label"] = plot_df["condition"].map(condition_labels)

    methods = sorted(plot_df["method"].unique())
    n_methods = len(methods)
    n_conditions = len(key_conditions)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(n_conditions)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        method_df = plot_df[plot_df["method"] == method]
        scores = []
        for cond in key_conditions:
            rows = method_df[method_df["condition"] == cond]
            scores.append(rows["auroc"].values[0] if len(rows) > 0 else np.nan)

        bars = ax.bar(
            x + i * width - (n_methods - 1) * width / 2,
            scores,
            width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#888888"),
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels
        for bar, score in zip(bars, scores):
            if not np.isnan(score):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{score:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([condition_labels[c] for c in key_conditions])
    ax.set_ylabel("AUROC")
    ax.set_title("Clean vs. Stressed Performance by Masking Method")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(bottom=max(0.5, plot_df["auroc"].min() - 0.05))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Figure 2: Performance Drop by Missing Modality ───────────────────────

def plot_modality_ablation_drop(df: pd.DataFrame, output_path: str):
    """Bar chart showing ΔAUROC when each modality is removed."""
    ablation_conditions = {
        "climate_missing": "Climate\ntimeseries",
        "satellite_missing": "Satellite\npatches",
        "tabular_missing": "Tabular\n(env vars)",
    }

    methods = sorted(df["method"].unique())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        method_df = df[df["method"] == method]
        clean_row = method_df[method_df["condition"] == "clean"]
        if len(clean_row) == 0:
            ax.set_title(f"{METHOD_LABELS.get(method, method)}\n(no data)")
            continue

        clean_auroc = clean_row["auroc"].values[0]
        drops = {}
        for cond, label in ablation_conditions.items():
            row = method_df[method_df["condition"] == cond]
            if len(row) > 0:
                drops[label] = clean_auroc - row["auroc"].values[0]

        labels = list(drops.keys())
        values = list(drops.values())
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        bars = ax.barh(labels, values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Δ AUROC (drop)")
        ax.set_title(METHOD_LABELS.get(method, method))

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center", fontsize=9,
            )

    fig.suptitle("Performance Drop When Modality is Missing", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Figure 3: Climate Noise Robustness Curve ─────────────────────────────

def plot_climate_noise_curve(df: pd.DataFrame, output_path: str):
    """Line plot: noise level vs AUROC for each method."""
    noise_conditions = {
        "clean": 0.0,
        "climate_noise_025": 0.25,
        "climate_noise_050": 0.50,
        "climate_noise_100": 1.00,
    }

    methods = sorted(df["method"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: absolute AUROC
    for method in methods:
        method_df = df[df["method"] == method]
        xs, ys = [], []
        for cond, noise_level in noise_conditions.items():
            rows = method_df[method_df["condition"] == cond]
            if len(rows) > 0:
                xs.append(noise_level)
                ys.append(rows["auroc"].values[0])

        if xs:
            ax1.plot(xs, ys, "o-", color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS.get(method, method), markersize=6,
                    linewidth=2)

    ax1.set_xlabel("Climate noise level (σ)")
    ax1.set_ylabel("AUROC")
    ax1.set_title("Absolute Performance")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Right: robustness ratio
    for method in methods:
        method_df = df[df["method"] == method]
        clean_row = method_df[method_df["condition"] == "clean"]
        if len(clean_row) == 0:
            continue
        clean_auroc = clean_row["auroc"].values[0]

        xs, ys = [], []
        for cond, noise_level in noise_conditions.items():
            rows = method_df[method_df["condition"] == cond]
            if len(rows) > 0:
                xs.append(noise_level)
                ys.append(rows["auroc"].values[0] / clean_auroc)

        if xs:
            ax2.plot(xs, ys, "s--", color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS.get(method, method), markersize=6,
                    linewidth=2)

    ax2.set_xlabel("Climate noise level (σ)")
    ax2.set_ylabel("Robustness ratio (AUROC / AUROC_clean)")
    ax2.set_title("Relative Robustness")
    ax2.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Climate Noise Robustness", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Figure 4: Month Dropout Robustness ───────────────────────────────────

def plot_month_dropout_curve(df: pd.DataFrame, output_path: str):
    """Line plot: fraction of climate months dropped vs AUROC."""
    drop_conditions = {
        "clean": 0.0,
        "climate_drop_025": 0.25,
        "climate_drop_050": 0.50,
        "climate_drop_075": 0.75,
    }

    methods = sorted(df["method"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in methods:
        method_df = df[df["method"] == method]
        xs, ys = [], []
        for cond, drop_frac in drop_conditions.items():
            rows = method_df[method_df["condition"] == cond]
            if len(rows) > 0:
                xs.append(drop_frac)
                ys.append(rows["auroc"].values[0])

        if xs:
            ax.plot(xs, ys, "o-", color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS.get(method, method), markersize=7,
                    linewidth=2)

    ax.set_xlabel("Fraction of climate months dropped")
    ax.set_ylabel("AUROC")
    ax.set_title("Robustness to Missing Climate Months")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75])
    ax.set_xticklabels(["0%", "25%", "50%", "75%"])

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Figure 5: Robustness Summary (heatmap) ───────────────────────────────

def plot_robustness_summary(df: pd.DataFrame, output_path: str):
    """Heatmap showing ΔAUROC for each method × condition."""
    display_conditions = [
        "climate_missing",
        "climate_noise_025",
        "climate_noise_050",
        "climate_noise_100",
        "climate_shift_p1",
        "climate_shift_p2",
        "climate_drop_025",
        "climate_drop_050",
        "climate_drop_075",
        "satellite_missing",
        "tabular_missing",
    ]

    condition_labels_short = {
        "climate_missing": "Climate\nmissing",
        "climate_noise_025": "Noise\nσ=0.25",
        "climate_noise_050": "Noise\nσ=0.50",
        "climate_noise_100": "Noise\nσ=1.00",
        "climate_shift_p1": "Shift\n+1σ",
        "climate_shift_p2": "Shift\n+2σ",
        "climate_drop_025": "Drop\n25% months",
        "climate_drop_050": "Drop\n50% months",
        "climate_drop_075": "Drop\n75% months",
        "satellite_missing": "Satellite\nmissing",
        "tabular_missing": "Tabular\nmissing",
    }

    methods = sorted(df["method"].unique())

    # Build matrix
    heatmap_data = {}
    for method in methods:
        method_df = df[df["method"] == method]
        clean_row = method_df[method_df["condition"] == "clean"]
        if len(clean_row) == 0:
            continue
        clean_auroc = clean_row["auroc"].values[0]
        row = []
        for cond in display_conditions:
            cond_rows = method_df[method_df["condition"] == cond]
            if len(cond_rows) > 0:
                row.append(clean_auroc - cond_rows["auroc"].values[0])
            else:
                row.append(np.nan)
        heatmap_data[method] = row

    if not heatmap_data:
        print("No data for summary heatmap")
        return

    data_matrix = np.array(list(heatmap_data.values()))
    method_names = list(heatmap_data.keys())
    cond_labels = [condition_labels_short[c] for c in display_conditions]

    fig, ax = plt.subplots(figsize=(12, max(3, len(methods) * 1.2)))

    im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=max(0, np.nanmin(data_matrix)),
                   vmax=np.nanmax(data_matrix))

    # Annotate cells
    for i in range(len(method_names)):
        for j in range(len(display_conditions)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.05 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                       fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(display_conditions)))
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in method_names], fontsize=10)
    ax.set_title("AUROC Drop by Method & Stress Condition\n(darker = more degradation)", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ AUROC", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Plot climate stress results")
    parser.add_argument("--results", type=str, default="results/climate_stress_results.csv",
                       help="Path to results CSV")
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="Output directory for figures")
    parser.add_argument("--format", type=str, default="png",
                       choices=["png", "pdf", "svg"],
                       help="Figure format")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        print("Run evaluate_climate_stress.py first.")
        sys.exit(1)

    df = load_results(args.results)
    print(f"Loaded {len(df)} results for {df['method'].nunique()} methods, "
          f"{df['condition'].nunique()} conditions")

    os.makedirs(args.output_dir, exist_ok=True)
    fmt = args.format

    # Generate all figures
    figures = [
        (plot_clean_vs_stress, "clean_vs_stress"),
        (plot_modality_ablation_drop, "modality_ablation_drop"),
        (plot_climate_noise_curve, "climate_noise_curve"),
        (plot_month_dropout_curve, "month_dropout_curve"),
        (plot_robustness_summary, "robustness_summary"),
    ]

    for plot_fn, name in figures:
        output_path = os.path.join(args.output_dir, f"{name}.{fmt}")
        try:
            plot_fn(df, output_path)
        except Exception as e:
            print(f"WARNING: Could not generate {name}: {e}")

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
