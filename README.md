# Climate-Modality Stress Testing for MIAM

Robustness of multimodal species distribution models under missing, noisy, and shifted climate inputs. Evaluates whether [MIAM](https://github.com/zbirobin/MIAM)'s imbalance-aware masking improves robustness when climate modalities are corrupted at test time.

## Setup

```bash
# Clone MIAM and install
git clone https://github.com/zbirobin/MIAM.git
cd MIAM && pip install -r requirements.txt && pip install -e . && cd ..

# Install additional dependencies
pip install huggingface_hub kagglehub seaborn

# Download pretrained checkpoints
hf download zbirobin/MIAM geoplant_miam.pt geoplant_dropout.pt geoplant_constant.pt
mkdir -p models/{miam,dropout,constant}
mv geoplant_miam.pt models/miam/miam.pt
mv geoplant_dropout.pt models/dropout/dropout.pt
mv geoplant_constant.pt models/constant/constant.pt

# Download GeoPlant PA dataset
python -c "import kagglehub; print(kagglehub.dataset_download('picekl/geoplant'))"
```

## Usage

```bash
# Run stress evaluation
python scripts/evaluate_climate_stress.py \
    --data_dir /path/to/geoplant \
    --checkpoints miam dropout constant

# Generate figures
python scripts/plot_climate_stress.py \
    --results results/climate_stress_results.csv \
    --output_dir figures
```

Alternatively, run cell by cell via `notebooks/colab_runner.py`.

## Structure

```
├── src/
│   └── stress_transforms.py           # Evaluation-time stress transforms
├── scripts/
│   ├── evaluate_climate_stress.py     # Evaluation loop
│   └── plot_climate_stress.py         # Figure generation
├── notebooks/
│   └── colab_runner.py               # Notebook script
├── results/                           # Output CSVs
└── figures/                           # Output plots
```

## Stress Conditions

| Category | Conditions |
|----------|-----------|
| Baseline | All modalities |
| Missing | Climate, satellite, or tabular modality removed |
| Noise | Gaussian noise on climate timeseries (sigma = 0.25, 0.50, 1.00) |
| Shift | Systematic climate shift (+1, +2 standard deviations) |
| Month dropout | 25%, 50%, or 75% of climate months zeroed |

## Data

[GeoPlant](https://plantnet.github.io/GeoPlant/) Presence-Absence subset (Kaggle: `picekl/geoplant`). Modalities include satellite image patches, Landsat time series, monthly climate time series, and environmental tabular variables.

## References

MIAM: Modality Imbalance-Aware Masking for Multimodal Ecological Applications. Zbinden et al., ICLR 2026.

GeoPlant: A Large-Scale Multimodal Dataset for Spatial Plant Species Prediction. Picek et al., NeurIPS 2024.
