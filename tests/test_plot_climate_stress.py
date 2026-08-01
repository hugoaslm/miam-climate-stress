import pandas as pd
import pytest

import plot_climate_stress

COLUMNS = ["method", "condition", "condition_group", "auroc", "auroc_std"]


def test_load_results_on_empty_csv(tmp_path):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=COLUMNS).to_csv(csv_path, index=False)
    df = plot_climate_stress.load_results(str(csv_path))
    assert len(df) == 0


def test_load_results_computes_robustness(tmp_path):
    rows = [
        {"method": "miam", "condition": "clean", "auroc": 0.9},
        {"method": "miam", "condition": "climate_missing", "auroc": 0.7},
    ]
    csv_path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = plot_climate_stress.load_results(str(csv_path))
    assert "auroc_drop" in df.columns
    assert df.loc[df["condition"] == "climate_missing", "auroc_drop"].iloc[0] == pytest.approx(0.2)


def test_load_results_missing_required_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame(columns=["method", "auroc"]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        plot_climate_stress.load_results(str(csv_path))
