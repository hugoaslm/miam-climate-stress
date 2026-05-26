"""
Climate-Modality Stress Testing for MIAM — Colab Runner
========================================================
Run this script cell-by-cell in Google Colab with an L4 GPU.

Total estimated time: 3-5 hours
GPU required: T4 or L4 (L4 recommended)
Disk: ~10 GB free
"""

# %% [CELL 1] — Environment Setup (5 min)
"""
Clone MIAM repo, install dependencies, download pretrained checkpoints.
"""

import os
import sys

# Clone MIAM
if not os.path.exists("MIAM"):
    !git clone https://github.com/zbirobin/MIAM.git

%cd MIAM

# Install dependencies
!pip install -q -r requirements.txt
!pip install -q -e .

# Install additional packages
!pip install -q huggingface_hub kagglehub jsonlines seaborn

# Download pretrained checkpoints
!hf download zbirobin/MIAM geoplant_miam.pt geoplant_dropout.pt geoplant_constant.pt

# Place checkpoints where evaluation expects them
!mkdir -p models/miam models/dropout models/constant
!mv geoplant_miam.pt models/miam/miam.pt
!mv geoplant_dropout.pt models/dropout/dropout.pt
!mv geoplant_constant.pt models/constant/constant.pt

print("✓ Setup complete. Checkpoints downloaded.")
!ls -la models/*/

# %% [CELL 2] — Download GeoPlant PA Dataset (30-60 min)
"""
Download the GeoPlant Presence-Absence dataset from Kaggle.
Requires: kaggle.json API key uploaded to Colab.
Alternative: upload from Google Drive if already downloaded.
"""

# Option A: Kaggle API
# First upload kaggle.json to /content/
# Then:
# !mkdir -p ~/.kaggle
# !cp /content/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

import kagglehub
print("Downloading GeoPlant dataset from Kaggle...")
print("This may take 30-60 minutes depending on connection speed.")
geo_path = kagglehub.dataset_download("picekl/geoplant")
print(f"✓ Dataset downloaded to: {geo_path}")

# The dataset structure:
# - PA_metadata_train.csv
# - EnvironmentalValues/ (Climate, Elevation, HumanFootprint, LandCover, SoilGrids)
# - SateliteTimeSeries-Landsat/values/ (6 band CSVs)
# - SateliteTimeSeries-Bioclimatic/values/ (monthly climate CSV)
# - SatellitePatches/ (RGB + NIR JPEGs zipped)

# Note: Satellite patches may need extraction from zips
# If available as pre-extracted, use directly. Otherwise:
# !unzip -q {geo_path}/SatellitePatches/patches_rgb.zip -d {geo_path}/SatellitePatches/PA-train-RGB/
# !unzip -q {geo_path}/SatellitePatches/patches_nir.zip -d {geo_path}/SatellitePatches/PA-train-NIR/

# Verify key files exist
import glob
key_files = [
    "PA_metadata_train.csv",
    "EnvironmentalValues/Climate/Average 1981-2010/PA-train-bioclimatic.csv",
]
for f in key_files:
    full = os.path.join(geo_path, f)
    if os.path.exists(full):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ MISSING: {f}")

# %% [CELL 2b] — Alternative: Skip download, use pre-existing results (0 min)
"""
FALLBACK: If data download is too slow or fails, skip to analysis
of existing MIAM results. These are already in the repo at:
  figures_tables/table_1/results/

We'll create a notebook cell below that works with these.
"""

# Skip to CELL 5 if using fallback.

# %% [CELL 3] — Set up stress test scripts (5 min)
"""
Copy our custom scripts into the MIAM repo.
"""

# Create directories
!mkdir -p ../src ../scripts ../results ../figures

# Write stress_transforms.py
# (content from src/stress_transforms.py — copy manually or via %%writefile)

# %% [CELL 4] — Run Stress Evaluations (30-90 min)
"""
Run the evaluation script for MIAM, Dropout, and Constant baselines.
"""

import sys
sys.path.insert(0, "..")

# Run evaluation
!python ../scripts/evaluate_climate_stress.py \
    --data_dir {geo_path} \
    --checkpoints miam dropout constant \
    --output_dir ../results \
    --batch_size 256

# Check results
import pandas as pd
df = pd.read_csv("../results/climate_stress_results.csv")
print(f"\nResults: {len(df)} rows")
print(df.head(20))

# %% [CELL 5] — Plot Results (15 min)
"""
Generate figures from evaluation results.
"""

!python ../scripts/plot_climate_stress.py \
    --results ../results/climate_stress_results.csv \
    --output_dir ../figures \
    --format png

# Display figures
from IPython.display import Image, display
import glob
for fig in sorted(glob.glob("../figures/*.png")):
    display(Image(filename=fig))
    print(f"--- {os.path.basename(fig)} ---")

# %% [CELL 6] — Summary Table
"""
Create a formatted summary table.
"""

import pandas as pd

df = pd.read_csv("../results/climate_stress_results.csv")

# Pivot: methods × conditions
pivot = df.pivot_table(
    values="auroc", index="condition", columns="method", aggfunc="first"
)

# Add robustness ratio
for method in pivot.columns:
    if "clean" in pivot.index:
        pivot[f"{method}_robustness"] = pivot[method] / pivot.loc["clean", method]

print("\n=== Full Results Table ===")
print(pivot.round(4).to_string())

pivot.round(4).to_csv("../results/climate_stress_summary.csv")
print("\n✓ Summary saved")

# %% [CELL 7] — Quick Stats for README
"""
Print key findings for the README.
"""

df = pd.read_csv("../results/climate_stress_results.csv")

for method in df["method"].unique():
    method_df = df[df["method"] == method]
    clean = method_df[method_df["condition"] == "clean"]["auroc"].values[0]

    # Climate missing drop
    climate_missing = method_df[method_df["condition"] == "climate_missing"]
    climate_drop = clean - climate_missing["auroc"].values[0] if len(climate_missing) else float("nan")

    # Climate noise (worst)
    noise_cols = [c for c in method_df["condition"].unique() if "noise" in c]
    noise_scores = method_df[method_df["condition"].isin(noise_cols)]["auroc"]

    print(f"\n{method}:")
    print(f"  Clean AUROC:            {clean:.4f}")
    print(f"  Climate missing drop:   {climate_drop:+.4f}")
    if len(noise_scores) > 0:
        print(f"  Worst noise (σ=1.0):    {noise_scores.min():.4f}")

print("\n---")
print("Key finding: MIAM should show lower degradation under climate stress")
print("compared to Dropout, despite possibly lower clean performance.")

# %% [CELL 8] — FALLBACK: Analysis using existing paper results
"""
CELL 8: Run this ONLY if data download failed (skip cells 2-7).

Uses pre-existing MIAM evaluation results from the repo to produce
modality contribution analysis without any new computation.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load existing results
results_dir = "figures_tables/table_1/results"
methods = ["miam", "modality_dropout", "constant"]

fallback_data = []
for method in methods:
    result_file = f"{results_dir}/{method}/all_together.jsonl"
    if os.path.exists(result_file):
        with open(result_file) as f:
            for line in f:
                row = json.loads(line)
                fallback_data.append({
                    "method": method.replace("modality_dropout", "dropout"),
                    "condition": row["variables"],
                    "auroc": row["test_auroc"],
                    "auroc_std": row["test_auroc_std"],
                })

fallback_df = pd.DataFrame(fallback_data)
print(f"Loaded {len(fallback_df)} existing results")
print(fallback_df.pivot_table(values="auroc", index="condition", columns="method"))

# Compute modality sensitivity
for method in fallback_df["method"].unique():
    method_df = fallback_df[fallback_df["method"] == method]
    all_score = method_df[method_df["condition"] == "all"]["auroc"].values[0]
    print(f"\n{method} (all={all_score:.4f}):")
    for _, row in method_df.iterrows():
        if row["condition"] != "all":
            sensitivity = all_score - row["auroc"]
            print(f"  {row['condition']:30s}: {row['auroc']:.4f} (sensitivity={sensitivity:+.4f})")

# Create modality sensitivity plot
fig, ax = plt.subplots(figsize=(10, 5))
conditions_plot = ["tabular", "timeseries", "climatic_timeseries", "sentinel2_patches"]
colors = {"miam": "#2ecc71", "dropout": "#e74c3c", "constant": "#3498db"}
x = np.arange(len(conditions_plot))
width = 0.25

for i, method in enumerate(["miam", "dropout", "constant"]):
    method_df = fallback_df[fallback_df["method"] == method]
    all_score = method_df[method_df["condition"] == "all"]["auroc"].values[0]
    scores = []
    for cond in conditions_plot:
        rows = method_df[method_df["condition"] == cond]
        scores.append(all_score - rows["auroc"].values[0] if len(rows) > 0 else np.nan)
    ax.bar(x + i * width, scores, width, label=method.upper(), color=colors[method])

ax.set_xticks(x + width)
ax.set_xticklabels([c.replace("_", "\n") for c in conditions_plot])
ax.set_ylabel("AUROC drop when modality removed")
ax.set_title("Modality Sensitivity by Masking Method (from paper results)")
ax.legend()
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/modality_sensitivity_fallback.png", dpi=150)
plt.show()

print("\n✓ Fallback analysis complete. Figure saved.")
print("Note: This uses static modality ablation, not dynamic stress tests.")
print("Full stress testing requires downloading the GeoPlant dataset.")

# %%
print("\n" + "="*60)
print("DONE! Check the figures/ and results/ directories.")
print("="*60)
