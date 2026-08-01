import torch


def _mask(batch, idx):
    batch = list(batch)
    batch[idx] = torch.full_like(batch[idx], float("nan"))
    return tuple(batch)


def mask_climate_modality(batch):
    return _mask(batch, 3)


def mask_satellite_modality(batch):
    return _mask(batch, 4)


def mask_tabular_modality(batch):
    return _mask(batch, 0)


def mask_landsat_modality(batch):
    return _mask(batch, 2)


def add_climate_noise(batch, sigma=0.25, seed=None):
    X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2 = batch
    if seed is not None:
        torch.manual_seed(seed)
    sample_std = X_climatic_ts.std(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-6)
    X_climatic_ts = X_climatic_ts + torch.randn_like(X_climatic_ts) * sigma * sample_std
    if seed is not None:
        torch.manual_seed(torch.initial_seed())
    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


def shift_climate(batch, n_stds=1.0, direction="positive"):
    X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2 = batch
    sample_std = X_climatic_ts.std(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-6)
    sign = 1.0 if direction == "positive" else -1.0
    X_climatic_ts = X_climatic_ts + sign * n_stds * sample_std
    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


def drop_climate_months(batch, fraction=0.5, seed=None):
    X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2 = batch
    if seed is not None:
        torch.manual_seed(seed)
    keep_mask = torch.rand(X_climatic_ts.shape, device=X_climatic_ts.device) > fraction
    X_climatic_ts = X_climatic_ts * keep_mask.float()
    if seed is not None:
        torch.manual_seed(torch.initial_seed())
    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


SEASON_MONTHS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}


def mask_climate_season(batch, season):
    X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2 = batch
    T = X_climatic_ts.shape[-1]
    months = SEASON_MONTHS[season]
    if T == 12:
        indices = [m - 1 for m in months]
    else:
        indices = list({(m - 1) * T // 12 for m in months})
    X_climatic_ts = X_climatic_ts.clone()
    for idx in indices:
        X_climatic_ts[..., idx] = 0.0
    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


TRANSFORM_REGISTRY = {
    "clean": None,
    "climate_missing": mask_climate_modality,
    "climate_noise_025": lambda b: add_climate_noise(b, sigma=0.25),
    "climate_noise_050": lambda b: add_climate_noise(b, sigma=0.50),
    "climate_noise_100": lambda b: add_climate_noise(b, sigma=1.00),
    "climate_shift_p1": lambda b: shift_climate(b, n_stds=1.0, direction="positive"),
    "climate_shift_p2": lambda b: shift_climate(b, n_stds=2.0, direction="positive"),
    "climate_shift_n1": lambda b: shift_climate(b, n_stds=1.0, direction="negative"),
    "climate_drop_025": lambda b: drop_climate_months(b, fraction=0.25),
    "climate_drop_050": lambda b: drop_climate_months(b, fraction=0.50),
    "climate_drop_075": lambda b: drop_climate_months(b, fraction=0.75),
    "climate_season_winter": lambda b: mask_climate_season(b, season="winter"),
    "climate_season_spring": lambda b: mask_climate_season(b, season="spring"),
    "climate_season_summer": lambda b: mask_climate_season(b, season="summer"),
    "climate_season_autumn": lambda b: mask_climate_season(b, season="autumn"),
    "satellite_missing": mask_satellite_modality,
    "tabular_missing": mask_tabular_modality,
    "landsat_missing": mask_landsat_modality,
}


def get_stress_condition_names():
    return list(TRANSFORM_REGISTRY.keys())


def apply_stress(batch, condition):
    transform = TRANSFORM_REGISTRY.get(condition)
    return batch if transform is None else transform(batch)


def get_condition_group(condition):
    if condition == "clean":
        return "Baseline"
    if condition.endswith("_missing"):
        return "Missing modality"
    if "noise" in condition:
        return "Climate noise"
    if "shift" in condition:
        return "Climate shift"
    if "drop" in condition:
        return "Month dropout"
    if "season" in condition:
        return "Seasonal mask"
    return "Other"
