"""
Evaluation script for climate-modality stress testing.

Loads pretrained MIAM checkpoints, evaluates under multiple stress
conditions, and saves results as CSV/JSON.

Usage:
    python scripts/evaluate_climate_stress.py \
        --data_dir /path/to/geoplant \
        --checkpoints miam dropout constant \
        --output_dir results/

Or from Colab:
    %run scripts/evaluate_climate_stress.py --data_dir /data/geoplant
"""

import os
import sys
import json
import argparse
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_auroc

# Add MIAM repo to path (assumes this script is in scripts/ under MIAM root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# We'll import from maskSDM after setup; wrapped in try for safety
try:
    from maskSDM.config import BASE_CONFIG, setup_device, add_data_specific_parameters
    from maskSDM.data.geoplant import get_geoplant_data, filter_species, split_data, normalize_data
    from maskSDM.training.helpers import create_dataloader, seed_everything
    from maskSDM.modules.model import get_model
    from maskSDM.modules.satellite_patches_tokenizers import get_sat_patch_tokenizer
    from maskSDM.evaluation.helpers import evaluate
except ImportError:
    print("ERROR: Could not import maskSDM. Make sure you're in the MIAM repo root.")
    print("Run: pip install -e .   from the MIAM repo")
    sys.exit(1)

# Import our stress transforms
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from stress_transforms import (
    TRANSFORM_REGISTRY,
    apply_stress,
    get_stress_condition_names,
    get_condition_group,
)


# ── Stress-adapted evaluate ──────────────────────────────────────────────

def evaluate_with_stress(
    config: Dict,
    model: torch.nn.Module,
    dataloader: DataLoader,
    token_names: List[str],
    used_token_names: List[str],
    stress_transform: Optional[callable] = None,
    tqdm_desc: str = "Evaluation",
) -> Tuple[float, float]:
    """
    Evaluate model with optional stress transform applied to each batch.

    Args:
        stress_transform: Function(batch) -> batch, or None for clean eval.

    Returns:
        (mean_auroc, std_auroc)
    """
    preds = []
    y_true = []

    for batch in tqdm.tqdm(dataloader, desc=tqdm_desc):
        (
            X_tabular_batch,
            y_batch,
            X_landsat_ts_batch,
            X_climatic_ts_batch,
            X_sentinel2_batch,
        ) = batch

        # Move to device
        X_tabular_batch = X_tabular_batch.to(config["device"])
        y_batch = y_batch.to(config["device"])
        X_landsat_ts_batch = X_landsat_ts_batch.to(config["device"])
        X_climatic_ts_batch = X_climatic_ts_batch.to(config["device"])
        X_sentinel2_batch = X_sentinel2_batch.to(config["device"])

        # Apply stress transform
        if stress_transform is not None:
            (
                X_tabular_batch,
                y_batch,
                X_landsat_ts_batch,
                X_climatic_ts_batch,
                X_sentinel2_batch,
            ) = stress_transform((
                X_tabular_batch,
                y_batch,
                X_landsat_ts_batch,
                X_climatic_ts_batch,
                X_sentinel2_batch,
            ))

        # Compute presence masks (same as original evaluate)
        X_tabular_mask = ~X_tabular_batch.isnan()

        X_landsat_ts_mask = (~X_landsat_ts_batch.isnan()).mean(
            dim=-1, dtype=torch.float
        ).reshape(X_landsat_ts_batch.shape[0], -1) == 1.0

        X_climatic_ts_mask = (~X_climatic_ts_batch.isnan()).mean(
            dim=-1, dtype=torch.float
        ).reshape(X_climatic_ts_batch.shape[0], -1) == 1.0

        X_sentinel2_mask = torch.ones(
            (X_sentinel2_batch.shape[0], config["sat_patch_mask_dim"]),
            device=config["device"],
        )

        X_mask = torch.cat(
            [
                X_tabular_mask,
                X_landsat_ts_mask,
                X_climatic_ts_mask,
                X_sentinel2_mask,
            ],
            axis=1,
        )

        # Mask out unused tokens
        not_used_tokens_idx = [
            token_names.index(tn)
            for tn in token_names
            if tn not in used_token_names
        ]
        X_mask[:, not_used_tokens_idx] = 0

        y_true.append(y_batch)

        y_pred = model(
            X_tabular_batch,
            X_landsat_ts_batch,
            X_climatic_ts_batch,
            X_sentinel2_batch,
            X_mask,
        )
        preds.append(y_pred)

    preds = torch.concatenate(preds, axis=0).float()
    y_true = torch.concatenate(y_true, axis=0).int()

    considered_species = config["indices_evaluated_species"]
    auc = binary_auroc(
        preds[:, considered_species].T,
        y_true[:, considered_species].T,
        num_tasks=len(considered_species),
    )

    return auc.mean().item(), auc.std().item()


# ── Main evaluation loop ─────────────────────────────────────────────────

def run_stress_evaluation(
    checkpoint_path: str,
    method_name: str,
    data_dir: str,
    conditions: List[str],
    base_config: Dict,
    device: torch.device,
) -> List[Dict]:
    """
    Run stress evaluation for a single checkpoint.

    Returns list of result dicts with keys:
        method, condition, condition_group, auroc, auroc_std
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Load and prepare data
    print("Loading data...")
    data = get_geoplant_data(data_path=data_dir)
    data = filter_species(data, min_num_obs=base_config["min_num_obs"])
    data = split_data(data)
    data = normalize_data(data)

    config = deepcopy(base_config)
    config["data_dir"] = data_dir
    config["device"] = device
    config = add_data_specific_parameters(config, data)

    # Token names (following existing pattern)
    token_names = (
        config["tabular_column_names"]
        + config["landsat_timeseries_column_names"]
        + config["climatic_timeseries_column_names"]
        + config["sentinel2_patches_column_names"]
    )

    # Create dataloader
    test_loader = create_dataloader(
        config, data, split="test", shuffle=False,
        batch_size=config["val_test_batch_size"]
    )

    # Load model
    sat_tokenizer_kwargs = {
        k.replace("get_sat_patch_tokenizer__", ""): v
        for k, v in config.items()
        if k.startswith("get_sat_patch_tokenizer__")
    }
    model = get_model(
        **config, sat_patch_tokenizer_kwargs=sat_tokenizer_kwargs
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    results = []
    with torch.no_grad():
        for condition in tqdm.tqdm(conditions, desc=f"{method_name} conditions"):
            stress_fn = TRANSFORM_REGISTRY.get(condition)

            auroc_mean, auroc_std = evaluate_with_stress(
                config=config,
                model=model,
                dataloader=test_loader,
                token_names=token_names,
                used_token_names=token_names,  # use all tokens
                stress_transform=stress_fn,
                tqdm_desc=f"{method_name}/{condition}",
            )

            results.append({
                "method": method_name,
                "condition": condition,
                "condition_group": get_condition_group(condition),
                "auroc": round(auroc_mean, 6),
                "auroc_std": round(auroc_std, 6),
            })

            print(f"  {condition}: AUROC = {auroc_mean:.4f} ± {auroc_std:.4f}")

    torch.cuda.empty_cache()
    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Climate-modality stress testing for MIAM"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to GeoPlant data directory"
    )
    parser.add_argument(
        "--checkpoints", type=str, nargs="+",
        default=["miam", "dropout", "constant"],
        help="Methods to evaluate"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="models",
        help="Directory containing method subdirectories with .pt files"
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+",
        default=None,
        help="Specific stress conditions to run (default: all)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory for output files"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base config
    base_config = deepcopy(BASE_CONFIG)
    base_config["device"] = device
    base_config["val_test_batch_size"] = args.batch_size
    base_config["data_dir"] = args.data_dir
    base_config["seed"] = args.seed

    # Which conditions to run
    if args.conditions:
        conditions = args.conditions
    else:
        conditions = [
            "clean",
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

    # Checkpoint name mapping
    checkpoint_files = {
        "miam": "miam.pt",
        "dropout": "dropout.pt",
        "constant": "constant.pt",
        "opm": "opm.pt",
        "dirichlet": "dirichlet.pt",
        "uniform": "uniform.pt",
    }

    all_results = []

    for method in args.checkpoints:
        ckpt_file = checkpoint_files.get(method, f"{method}.pt")
        ckpt_path = os.path.join(args.checkpoint_dir, method, ckpt_file)

        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            print("  Download from: hf download zbirobin/MIAM geoplant_{method}.pt")
            continue

        try:
            method_results = run_stress_evaluation(
                checkpoint_path=ckpt_path,
                method_name=method,
                data_dir=args.data_dir,
                conditions=conditions,
                base_config=base_config,
                device=device,
            )
            all_results.extend(method_results)
        except Exception as e:
            print(f"ERROR evaluating {method}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "climate_stress_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # JSON
    json_path = os.path.join(args.output_dir, "climate_stress_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Quick summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method in args.checkpoints:
        method_df = df[df["method"] == method]
        if len(method_df) == 0:
            continue
        clean_auroc = method_df[method_df["condition"] == "clean"]["auroc"].values
        clean_str = f"{clean_auroc[0]:.4f}" if len(clean_auroc) > 0 else "N/A"
        print(f"\n{method}:")
        print(f"  Clean: {clean_str}")
        for _, row in method_df.iterrows():
            if row["condition"] != "clean":
                drop = clean_auroc[0] - row["auroc"] if len(clean_auroc) > 0 else float("nan")
                print(f"  {row['condition']:30s}: {row['auroc']:.4f} (Δ={drop:+.4f})")


if __name__ == "__main__":
    main()
