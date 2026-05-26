"""
Stress-test transforms for climate modality in GeoPlant/MIAM evaluation.

Each transform takes a batch and returns a modified batch suitable for
evaluation-time stress testing. All transforms operate on the numpy arrays
or torch tensors that constitute a GeoPlant batch:

    (X_tabular, y, X_landsat_timeseries, X_climatic_timeseries, X_sentinel2_patches)

The climatic_timeseries has shape: (batch, n_channels, n_years, n_timepoints)
where:
  - n_channels = 4 (tmean, tmin, tmax, precip)
  - n_timepoints = 12 (monthly) or reduced resolution
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Literal


# ── Climate masking ──────────────────────────────────────────────────────

def mask_climate_modality(
    batch: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    """
    Mask the entire climate timeseries modality by setting values to NaN.
    MIAM's evaluation pipeline detects NaN → sets mask=0 → replaces with
    learned mask_token. This is true modality-level masking.
    """
    (
        X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2,
    ) = batch
    X_climatic_ts = torch.full_like(X_climatic_ts, float("nan"))
    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


def mask_satellite_modality(
    batch: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    """Mask Sentinel-2 satellite patches modality (set to NaN)."""
    (
        X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2,
    ) = batch
    X_sentinel2 = torch.full_like(X_sentinel2, float("nan"))
    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


def mask_tabular_modality(
    batch: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    """Mask the tabular/environmental modality (set to NaN)."""
    (
        X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2,
    ) = batch
    X_tabular = torch.full_like(X_tabular, float("nan"))
    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


def mask_landsat_modality(
    batch: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    """Mask the Landsat timeseries modality (set to NaN)."""
    (
        X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2,
    ) = batch
    X_landsat_ts = torch.full_like(X_landsat_ts, float("nan"))
    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


# ── Climate noise ────────────────────────────────────────────────────────

def add_climate_noise(
    batch: Tuple[torch.Tensor, ...],
    sigma: float = 0.25,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Add Gaussian noise to climate timeseries.
    Noise is relative to the per-sample standard deviation.

    Args:
        sigma: Noise standard deviation (as fraction of per-sample std).
               σ=0.25 means noise ~ N(0, 0.25² × sample_var)
        seed: Random seed for reproducibility.
    """
    (
        X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2,
    ) = batch

    if seed is not None:
        torch.manual_seed(seed)

    # Compute per-sample std (avoid division by zero)
    sample_std = X_climatic_ts.std(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-6)
    noise = torch.randn_like(X_climatic_ts) * sigma * sample_std
    X_climatic_ts = X_climatic_ts + noise

    if seed is not None:
        torch.manual_seed(torch.initial_seed())  # reset

    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


# ── Climate shift ────────────────────────────────────────────────────────

def shift_climate(
    batch: Tuple[torch.Tensor, ...],
    n_stds: float = 1.0,
    direction: Literal["positive", "negative"] = "positive",
) -> Tuple[torch.Tensor, ...]:
    """
    Add a systematic shift to climate timeseries.
    Shift is applied as n_stds × per-sample standard deviation.

    Args:
        n_stds: Number of standard deviations to shift.
        direction: Direction of shift.
    """
    (
        X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2,
    ) = batch

    sample_std = X_climatic_ts.std(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-6)
    sign = 1.0 if direction == "positive" else -1.0
    X_climatic_ts = X_climatic_ts + sign * n_stds * sample_std

    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


# ── Climate month dropout ───────────────────────────────────────────────

def drop_climate_months(
    batch: Tuple[torch.Tensor, ...],
    fraction: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Randomly zero out a fraction of climate months (within-modality corruption).
    Unlike NaN masking, zeros go through the tokenizer normally — this tests
    robustness to partial within-modality information loss.

    Operates on the last dimension (timepoints/months).

    Args:
        fraction: Fraction of months to drop (0.0 to 1.0).
        seed: Random seed.
    """
    (
        X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2,
    ) = batch

    if seed is not None:
        torch.manual_seed(seed)

    # Shape: (B, C, Y, T) — mask along the last dim (T)
    B, C, Y, T = X_climatic_ts.shape
    keep_mask = torch.rand(B, C, Y, T, device=X_climatic_ts.device) > fraction
    X_climatic_ts = X_climatic_ts * keep_mask.float()

    if seed is not None:
        torch.manual_seed(torch.initial_seed())

    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


# ── Seasonal climate masking ────────────────────────────────────────────

SEASON_MONTHS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}


def mask_climate_season(
    batch: Tuple[torch.Tensor, ...],
    season: Literal["winter", "spring", "summer", "autumn"],
) -> Tuple[torch.Tensor, ...]:
    """
    Zero out climate data for a specific season (within-modality corruption).
    Assume last dim = months (12 for full resolution, 1-indexed).
    Zeros go through tokenizer — tests seasonal information contribution.

    Args:
        season: One of 'winter', 'spring', 'summer', 'autumn'.
    """
    (
        X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2,
    ) = batch

    months_to_mask = SEASON_MONTHS[season]
    T = X_climatic_ts.shape[-1]

    # If T != 12, we approximate by mapping to the full 12-month year
    # and masking proportionally
    if T != 12:
        # Map 12-month indices to available timepoints
        month_indices = [(m - 1) * T // 12 for m in months_to_mask]
        month_indices = list(set(month_indices))  # deduplicate
    else:
        month_indices = [m - 1 for m in months_to_mask]  # 0-indexed

    X_climatic_ts = X_climatic_ts.clone()
    for idx in month_indices:
        X_climatic_ts[..., idx] = 0.0

    return X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2


# ── Apply transform by name ─────────────────────────────────────────────

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


def get_stress_condition_names() -> list:
    """Return all available stress condition names."""
    return list(TRANSFORM_REGISTRY.keys())


def apply_stress(
    batch: Tuple[torch.Tensor, ...],
    condition: str,
) -> Tuple[torch.Tensor, ...]:
    """
    Apply a named stress condition to a batch.

    Args:
        batch: Tuple of (X_tabular, y, X_landsat_ts, X_climatic_ts, X_sentinel2).
        condition: Name from TRANSFORM_REGISTRY.

    Returns:
        Modified batch (or original batch if condition is 'clean').
    """
    transform = TRANSFORM_REGISTRY.get(condition)
    if transform is None:
        return batch
    return transform(batch)


def get_condition_group(condition: str) -> str:
    """Return the group category for a condition (for plotting)."""
    if condition == "clean":
        return "Baseline"
    if "missing" in condition or condition.endswith("_missing"):
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
