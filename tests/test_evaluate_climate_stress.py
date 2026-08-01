import os

import pandas as pd
import pytest

from conftest import import_evaluate_climate_stress


class _Args:
    def __init__(self, ckpt_dir, out_dir, checkpoints):
        self.data_dir = "/tmp/fake_data"
        self.checkpoints = checkpoints
        self.checkpoint_dir = ckpt_dir
        self.conditions = None
        self.output_dir = out_dir
        self.no_satellite = False
        self.batch_size = 4
        self.seed = 42


class TestEmptyResultsGuard:
    def test_missing_checkpoints_no_crash(self, tmp_path, monkeypatch, capsys):
        ev = import_evaluate_climate_stress()

        ckpt_dir = tmp_path / "models"
        ckpt_dir.mkdir()
        out_dir = tmp_path / "results"

        monkeypatch.setattr(
            ev, "parse_args", lambda: _Args(str(ckpt_dir), str(out_dir), ["miam", "dropout", "constant"])
        )
        ev.main()

        csv_path = out_dir / "climate_stress_results.csv"
        json_path = out_dir / "climate_stress_results.json"
        assert csv_path.exists()
        assert json_path.exists()
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["method", "condition", "condition_group", "auroc", "auroc_std"]
        assert len(df) == 0
        assert "No results were produced" in capsys.readouterr().out

    def test_summary_with_results(self, tmp_path, monkeypatch):
        ev = import_evaluate_climate_stress()

        ckpt_dir = tmp_path / "models"
        os.makedirs(os.path.join(ckpt_dir, "miam"))
        with open(os.path.join(ckpt_dir, "miam", "miam.pt"), "wb") as f:
            f.write(b"fake")

        out_dir = tmp_path / "results"

        monkeypatch.setattr(ev, "parse_args", lambda: _Args(str(ckpt_dir), str(out_dir), ["miam"]))
        monkeypatch.setattr(
            ev,
            "run_stress_evaluation",
            lambda **kw: [
                {"method": "miam", "condition": "clean", "condition_group": "Baseline", "auroc": 0.9, "auroc_std": 0.01},
                {"method": "miam", "condition": "climate_missing", "condition_group": "Missing modality", "auroc": 0.7, "auroc_std": 0.01},
            ],
        )
        ev.main()

        df = pd.read_csv(out_dir / "climate_stress_results.csv")
        assert len(df) == 2
        assert set(df["method"]) == {"miam"}
        assert set(df["condition"]) == {"clean", "climate_missing"}


class TestDeterministicCuda:
    def test_cublas_workspace_config_set(self):
        import_evaluate_climate_stress()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8")
