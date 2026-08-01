import pytest
import torch

from stress_transforms import (
    TRANSFORM_REGISTRY,
    apply_stress,
    get_condition_group,
    get_stress_condition_names,
)

B = 4
DTYPE = torch.float32


def make_batch():
    return (
        torch.randn(B, 32, dtype=DTYPE),
        torch.randint(0, 2, (B,), dtype=torch.int64),
        torch.randn(B, 6, 24, dtype=DTYPE),
        torch.randn(B, 8, 12, dtype=DTYPE),
        torch.randn(B, 4, 4, 8, 8, dtype=DTYPE),
    )


def test_registry_names():
    expected = {
        "clean",
        "climate_missing",
        "climate_noise_025",
        "climate_noise_050",
        "climate_noise_100",
        "climate_shift_p1",
        "climate_shift_p2",
        "climate_shift_n1",
        "climate_drop_025",
        "climate_drop_050",
        "climate_drop_075",
        "climate_season_winter",
        "climate_season_spring",
        "climate_season_summer",
        "climate_season_autumn",
        "satellite_missing",
        "tabular_missing",
        "landsat_missing",
    }
    assert set(get_stress_condition_names()) == expected


def test_clean_is_identity():
    batch = make_batch()
    assert apply_stress(batch, "clean") is batch


@pytest.mark.parametrize("name", [n for n in TRANSFORM_REGISTRY if n != "clean"])
def test_shape_and_dtype_preserved(name):
    batch = make_batch()
    out = apply_stress(batch, name)
    assert isinstance(out, tuple) and len(out) == 5
    for orig, new in zip(batch, out):
        assert new.shape == orig.shape
        assert new.dtype == orig.dtype


@pytest.mark.parametrize(
    "name,idx",
    [
        ("tabular_missing", 0),
        ("landsat_missing", 2),
        ("climate_missing", 3),
        ("satellite_missing", 4),
    ],
)
def test_missing_modality_sets_nan(name, idx):
    batch = make_batch()
    out = apply_stress(batch, name)
    assert bool(torch.isnan(out[idx]).all())
    for j, t in enumerate(batch):
        if j != idx:
            assert torch.equal(out[j], t)


@pytest.mark.parametrize(
    "name,sigma",
    [
        ("climate_noise_025", 0.25),
        ("climate_noise_050", 0.50),
        ("climate_noise_100", 1.00),
    ],
)
def test_noise_adds_noise_keeps_finite(name, sigma):
    torch.manual_seed(0)
    batch = make_batch()
    out = apply_stress(batch, name)
    X_clim_in = batch[3]
    X_clim_out = out[3]
    assert not torch.equal(X_clim_out, X_clim_in)
    assert bool(torch.isfinite(X_clim_out).all())
    sample_std = X_clim_in.std(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-6)
    diff = (X_clim_out - X_clim_in) / sample_std
    assert torch.allclose(diff.std().cpu(), torch.tensor(sigma, dtype=DTYPE), atol=0.2)
    for j, t in enumerate(batch):
        if j != 3:
            assert torch.equal(out[j], t)


@pytest.mark.parametrize(
    "name,direction",
    [
        ("climate_shift_p1", 1.0),
        ("climate_shift_p2", 2.0),
        ("climate_shift_n1", -1.0),
    ],
)
def test_shift_magnitude_and_direction(name, direction):
    batch = make_batch()
    out = apply_stress(batch, name)
    X_in = batch[3]
    X_out = out[3]
    sample_std = X_in.std(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-6)
    diff = X_out - X_in
    assert torch.allclose(diff, direction * sample_std, atol=1e-5)
    if direction < 0:
        assert bool((diff < 0).all())
    else:
        assert bool((diff >= 0).all())


@pytest.mark.parametrize(
    "name,fraction",
    [
        ("climate_drop_025", 0.25),
        ("climate_drop_050", 0.50),
        ("climate_drop_075", 0.75),
    ],
)
def test_drop_zeroes_fraction(name, fraction):
    batch = make_batch()
    torch.manual_seed(0)
    out = apply_stress(batch, name)
    X_in = batch[3]
    X_out = out[3]
    zeroed = X_out == 0.0
    kept = (X_in != 0.0) & (X_out != 0.0)
    assert float(zeroed.float().mean()) == pytest.approx(fraction, abs=0.1)
    assert bool((X_out[kept] == X_in[kept]).all())


@pytest.mark.parametrize(
    "season,months",
    [
        ("winter", [0, 11, 1]),
        ("spring", [2, 3, 4]),
        ("summer", [5, 6, 7]),
        ("autumn", [8, 9, 10]),
    ],
)
def test_season_mask_zeroes_months(season, months):
    batch = make_batch()
    out = apply_stress(batch, f"climate_season_{season}")
    X_in = batch[3]
    X_out = out[3]
    assert bool((X_out[..., months] == 0.0).all())
    other = [t for t in range(12) if t not in months]
    assert torch.equal(X_out[..., other], X_in[..., other])
    assert torch.equal(X_in, batch[3])


@pytest.mark.parametrize(
    "condition,group",
    [
        ("clean", "Baseline"),
        ("climate_missing", "Missing modality"),
        ("satellite_missing", "Missing modality"),
        ("tabular_missing", "Missing modality"),
        ("landsat_missing", "Missing modality"),
        ("climate_noise_025", "Climate noise"),
        ("climate_shift_p1", "Climate shift"),
        ("climate_drop_050", "Month dropout"),
        ("climate_season_winter", "Seasonal mask"),
    ],
)
def test_condition_grouping(condition, group):
    assert get_condition_group(condition) == group
