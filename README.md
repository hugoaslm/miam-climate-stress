# Climate-Modality Stress Testing for MIAM

This repository studies the robustness of multimodal species distribution models to degradation of the climate-modality input at test time. It assesses whether the modality imbalance-aware masking strategy proposed in MIAM (Zbinden et al., 2026) confers an advantage over standard masking baselines when the climatic time series is missing, corrupted, or systematically shifted relative to training conditions. Experiments are conducted on the GeoPlant Presence–Absence benchmark (Picek et al., 2024) using the released MIAM, Modality Dropout, and Constant Masking checkpoints.

## Method

Each checkpoint is evaluated on the held-out test split under a set of stress conditions, applied to every batch prior to inference. Performance is reported as the mean AUROC over the evaluated species, together with the standard deviation.

| Category | Conditions | Description |
|----------|------------|-------------|
| Baseline | `clean` | All modalities present |
| Missing modality | `climate_missing`, `satellite_missing`, `tabular_missing`, `landsat_missing` | The specified modality is fully masked (NaN inputs mapped to the learned mask token) |
| Climate noise | `climate_noise_025`, `climate_noise_050`, `climate_noise_100` | Additive Gaussian noise, σ = 0.25, 0.50, 1.00 relative to the per-sample variance |
| Climate shift | `climate_shift_p1`, `climate_shift_p2`, `climate_shift_n1` | Systematic shift by ±1, ±2 standard deviations |
| Month dropout | `climate_drop_025`, `climate_drop_050`, `climate_drop_075` | A random 25%, 50%, or 75% of climate months zeroed |
| Seasonal mask | `climate_season_winter`, `climate_season_spring`, `climate_season_summer`, `climate_season_autumn` | Climate values zeroed for one season |

The full GeoPlant distribution is approximately 44 GB, of which roughly 40 GB corresponds to satellite patches. In the default configuration (`--no_satellite`) the satellite modality is kept permanently masked, which avoids downloading the patch files; in this mode the `satellite_missing` condition is omitted and clean AUROC values differ from those reported in the paper.

## Data

GeoPlant Presence–Absence subset (Kaggle: `picekl/geoplant`). Four input modalities are used: environmental tabular variables, Landsat time series, monthly climate time series, and Sentinel-2 satellite patches.

## Usage

### Installation

```bash
git clone https://github.com/zbirobin/MIAM.git
cd MIAM && pip install -r requirements.txt && pip install -e . && cd ..

pip install "huggingface_hub[cli]" kagglehub torcheval
```

Download the pretrained checkpoints:

```bash
hf download zbirobin/MIAM geoplant_miam.pt geoplant_dropout.pt geoplant_constant.pt
mkdir -p models/{miam,dropout,constant}
mv geoplant_miam.pt models/miam/miam.pt
mv geoplant_dropout.pt models/dropout/dropout.pt
mv geoplant_constant.pt models/constant/constant.pt
```

Download the GeoPlant CSV files required by the default configuration:

```bash
python -c "
import kagglehub, os
files = [
    'PA_metadata_train.csv',
    'EnvironmentalValues/Climate/Average 1981-2010/PA-train-bioclimatic.csv',
    'EnvironmentalValues/Elevation/PA-train-elevation.csv',
    'EnvironmentalValues/Human Footprint/PA-train-human_footprint.csv',
    'EnvironmentalValues/LandCover/PA-train-landcover.csv',
    'EnvironmentalValues/SoilGrids/PA-train-soilgrids.csv',
    'SateliteTimeSeries-Bioclimatic/values/PA-train-bioclimatic-monthly.csv',
] + [
    f'SateliteTimeSeries-Landsat/values/PA-train-landsat_time_series/PA-train-landsat_time_series-{b}.csv'
    for b in ('red', 'green', 'blue', 'nir', 'swir1', 'swir2')
]
paths = [kagglehub.dataset_download('picekl/geoplant', path=f) for f in files]
print(os.path.dirname(paths[0]))
"
```

### Evaluation

```bash
python scripts/evaluate_climate_stress.py \
    --data_dir /path/to/geoplant \
    --checkpoints miam dropout constant \
    --no_satellite
```

Results are written to `results/climate_stress_results.csv` and `.json`.

```bash
python scripts/plot_climate_stress.py \
    --results results/climate_stress_results.csv \
    --output_dir figures
```

Alternatively, `notebooks/miam_climate_stress.ipynb` reproduces the full pipeline end-to-end in Google Colab.

## Repository Layout

```
├── src/
│   └── stress_transforms.py           Evaluation-time stress transforms
├── scripts/
│   ├── evaluate_climate_stress.py     Evaluation loop
│   └── plot_climate_stress.py         Figure generation
├── notebooks/
│   └── miam_climate_stress.ipynb      Colab reproduction script
├── results/                           Output tables
└── figures/                           Output figures
```

## References

```bibtex
@inproceedings{zbinden2026miam,
  title={MIAM: Modality Imbalance-Aware Masking for Multimodal Ecological Applications},
  author={Robin Zbinden and Wesley Monteith-Finas and Gencer Sumbul and Nina van Tiel and Chiara Vanalli and Devis Tuia},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}

@inproceedings{picek2024geoplant,
  title={GeoPlant: A Large-Scale Multimodal Dataset for Spatial Plant Species Prediction},
  author={Picek, Lukas and Botella, Christophe and Servajean, Maximilien and Leblanc, Cesar and Palard, Remi and Larcher, Theo and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
