from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


R2_LABELS = {
    "r2_home_fpn48_lr0750_seed2_100e": "Home FPN48 LR0.00075 seed2 (partial)",
    "r2_school1_fpn48_lr1000_seed2": "S1 FPN48 LR0.001 seed2",
    "r2_school2_fpn48_lr0750": "S2 FPN48 LR0.00075",
    "r2_school3_fpn48_lr0500": "S3 FPN48 LR0.0005",
    "r2_school4_fpn40_lr1000": "S4 FPN40 LR0.001",
    "r2_school5_fpn56_lr1000": "S5 FPN56 LR0.001",
    "r2_school6_fpn48_exp250": "S6 FPN48 exp2.5 LR0.001",
}

R1_NAMES = {
    "home_baseline_100e",
    "school1_low_lr_100e",
    "school2_lightweight_100e",
    "school3_capacity_100e",
    "school4_box_weight_100e",
    "school5_center_radius_100e",
}


def decode(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")


def load_archives(source: Path) -> dict[str, dict]:
    experiments: dict[str, dict] = {}
    for archive in sorted(source.glob("results*.zip")):
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            histories = [name for name in names if name.endswith("/history.csv") and "/training/" in name]
            for history_name in histories:
                parts = history_name.replace("\\", "/").split("/")
                experiment = parts[-2]
                if experiment not in R2_LABELS and experiment not in R1_NAMES:
                    continue
                # Prefer the archive containing the Round 2 experiment and avoid
                # overwriting a longer duplicate historical run.
                frame = pd.read_csv(io.BytesIO(zf.read(history_name)))
                current = experiments.get(experiment)
                if current is not None and len(current["history"]) >= len(frame):
                    continue
                base = history_name.rsplit("/", 1)[0]
                config_name = f"{base}/config.json"
                device_name = f"{base}/device.json"
                log_name = f"{base}/runner_console.log"
                warnings_name = f"{base}/warnings.log"
                experiments[experiment] = {
                    "archive": archive,
                    "base": base,
                    "history": frame,
                    "config": json.loads(decode(zf.read(config_name))) if config_name in names else {},
                    "device": json.loads(decode(zf.read(device_name))) if device_name in names else {},
                    "runner_log": decode(zf.read(log_name)) if log_name in names else "",
                    "warnings_log": decode(zf.read(warnings_name)) if warnings_name in names else "",
                    "best_exists": f"{base}/checkpoints/best.pt" in names,
                    "last_exists": f"{base}/checkpoints/last.pt" in names,
                    "best_bytes": zf.getinfo(f"{base}/checkpoints/best.pt").file_size
                    if f"{base}/checkpoints/best.pt" in names else 0,
                }
    return experiments


def finite_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.max()) if len(values) else math.nan


def summarize(name: str, item: dict, cutoff: int | None = None) -> dict:
    full = item["history"].copy()
    frame = full if cutoff is None else full[full["epoch"] <= cutoff].copy()
    best = frame.loc[frame["metric_map50_95"].idxmax()]
    last = frame.loc[frame["epoch"].idxmax()]
    best_f1_row = frame.loc[frame["metric_best_f1"].idxmax()]
    last10 = frame.tail(min(10, len(frame)))
    log = item["runner_log"]
    attempts = len(re.findall(r"attempt=\d+", log))
    return_codes = [int(value) for value in re.findall(r"return_code=(-?\d+)", log)]
    warnings = int(pd.to_numeric(frame["warning_count"], errors="coerce").fillna(0).sum())
    gradients = pd.to_numeric(frame["train_gradient_norm"], errors="coerce")
    nonfinite_gradients = int((~np.isfinite(gradients)).sum())

    config = item["config"]
    device = item["device"]
    return {
        "experiment": name,
        "label": R2_LABELS.get(name, name),
        "archive": item["archive"].name,
        "cutoff": cutoff or int(full["epoch"].max()),
        "planned_epochs": int(config.get("epochs", 0)),
        "last_epoch": int(last["epoch"]),
        "best_epoch": int(best["epoch"]),
        "map50_95": float(best["metric_map50_95"]),
        "ap50": float(best["metric_ap50"]),
        "ap75": float(best["metric_ap75"]),
        "precision_at_025": float(best["metric_precision"]),
        "recall_at_025": float(best["metric_recall"]),
        "f1_at_025": float(best["metric_f1"]),
        "best_f1_at_map_epoch": float(best["metric_best_f1"]),
        "best_f1_threshold_at_map_epoch": float(best["metric_best_f1_threshold"]),
        "tiny_recall": float(best["metric_tiny_lt16_recall"]),
        "small_recall": float(best["metric_small_16_32_recall"]),
        "medium_recall": float(best["metric_medium_32_96_recall"]),
        "large_recall": float(best["metric_large_ge96_recall"]),
        "valid_total": float(best["valid_total"]),
        "train_total": float(best["train_total"]),
        "generalization_gap": float(best["valid_total"] - best["train_total"]),
        "peak_f1_any_epoch": float(best_f1_row["metric_best_f1"]),
        "peak_f1_epoch": int(best_f1_row["epoch"]),
        "peak_f1_threshold": float(best_f1_row["metric_best_f1_threshold"]),
        "last_map50_95": float(last["metric_map50_95"]),
        "last_f1_at_025": float(last["metric_f1"]),
        "last_valid_total": float(last["valid_total"]),
        "last10_map_mean": float(last10["metric_map50_95"].mean()),
        "last10_map_std": float(last10["metric_map50_95"].std(ddof=0)),
        "map_drop_best_to_last": float(best["metric_map50_95"] - last["metric_map50_95"]),
        "warning_count_sum": warnings,
        "nonfinite_gradient_epochs": nonfinite_gradients,
        "max_finite_gradient_norm": finite_max(gradients),
        "median_images_per_second": float(frame["train_images_per_second"].median()),
        "peak_vram_gib": float(frame["peak_vram_gib"].max()),
        "total_hours": float(frame["seconds"].sum() / 3600.0),
        "attempts": attempts,
        "return_codes": ",".join(map(str, return_codes)),
        "runner_completed": bool(return_codes and return_codes[-1] == 0),
        "best_exists": item["best_exists"],
        "last_exists": item["last_exists"],
        "checkpoint_bytes": item["best_bytes"],
        "seed": int(config.get("seed", 0)),
        "learning_rate": float(config.get("learning_rate", 0)),
        "fpn_channels": int(config.get("fpn_channels", 0)),
        "backbone_expansion": float(config.get("backbone_expansion", 0)),
        "amp_initial_scale": float(config.get("amp_initial_scale", 0)),
        "amp_growth_interval": int(config.get("amp_growth_interval", 0)),
        "torch_version": str(device.get("torch_version", "")),
        "cuda_build": str(device.get("cuda_build", "")),
        "gpu_name": str(device.get("gpu_name", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Folder containing results*.zip archives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("round2_analysis"),
        help="Output folder for CSV summaries",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    experiments = load_archives(source)
    missing = set(R2_LABELS) - set(experiments)
    if missing:
        raise RuntimeError(f"Missing Round 2 experiments: {sorted(missing)}")

    r2_full = pd.DataFrame([summarize(name, experiments[name]) for name in R2_LABELS])
    r2_common = pd.DataFrame([summarize(name, experiments[name], cutoff=50) for name in R2_LABELS])

    epoch50_rows = []
    for name in R2_LABELS:
        item = experiments[name]
        row = item["history"].loc[item["history"]["epoch"] == 50].iloc[0]
        epoch50_rows.append({
            "experiment": name,
            "label": R2_LABELS[name],
            "map50_95": float(row["metric_map50_95"]),
            "precision_at_025": float(row["metric_precision"]),
            "recall_at_025": float(row["metric_recall"]),
            "f1_at_025": float(row["metric_f1"]),
            "best_f1": float(row["metric_best_f1"]),
            "best_f1_threshold": float(row["metric_best_f1_threshold"]),
            "tiny_recall": float(row["metric_tiny_lt16_recall"]),
            "small_recall": float(row["metric_small_16_32_recall"]),
            "valid_total": float(row["valid_total"]),
        })
    epoch50 = pd.DataFrame(epoch50_rows)

    r1 = pd.DataFrame([
        summarize(name, experiments[name]) for name in sorted(R1_NAMES) if name in experiments
    ])

    r2_full.sort_values("map50_95", ascending=False).to_csv(output / "r2_full_summary.csv", index=False)
    r2_common.sort_values("map50_95", ascending=False).to_csv(output / "r2_best_through_epoch50.csv", index=False)
    epoch50.sort_values("map50_95", ascending=False).to_csv(output / "r2_epoch50_snapshot.csv", index=False)
    r1.sort_values("map50_95", ascending=False).to_csv(output / "r1_reconstructed_summary.csv", index=False)

    curve_rows = []
    for name in R2_LABELS:
        frame = experiments[name]["history"].copy()
        frame.insert(0, "experiment", name)
        frame.insert(1, "label", R2_LABELS[name])
        curve_rows.append(frame)
    pd.concat(curve_rows, ignore_index=True).to_csv(output / "r2_all_histories.csv", index=False)

    print("\n=== ROUND 2 FULL/PARTIAL BEST ===")
    cols = [
        "label", "last_epoch", "best_epoch", "map50_95", "precision_at_025",
        "recall_at_025", "f1_at_025", "best_f1_at_map_epoch",
        "best_f1_threshold_at_map_epoch", "tiny_recall", "small_recall",
        "last10_map_mean", "map_drop_best_to_last", "warning_count_sum",
        "attempts", "return_codes", "median_images_per_second", "peak_vram_gib",
    ]
    print(r2_full.sort_values("map50_95", ascending=False)[cols].to_string(index=False))

    print("\n=== FAIR BEST THROUGH EPOCH 50 ===")
    print(r2_common.sort_values("map50_95", ascending=False)[cols[:13]].to_string(index=False))

    print("\n=== EXACT EPOCH 50 ===")
    print(epoch50.sort_values("map50_95", ascending=False).to_string(index=False))

    print("\n=== ROUND 1 RECONSTRUCTED ===")
    print(r1.sort_values("map50_95", ascending=False)[["experiment", "last_epoch", "best_epoch", "map50_95", "recall_at_025", "f1_at_025", "tiny_recall", "small_recall"]].to_string(index=False))

    print("\n=== DEVICE/CONFIG ===")
    print(r2_full[["label", "seed", "learning_rate", "fpn_channels", "backbone_expansion", "torch_version", "cuda_build", "gpu_name", "checkpoint_bytes"]].to_string(index=False))

    print("\n=== LOG TAILS ===")
    for name in R2_LABELS:
        item = experiments[name]
        lines = [line for line in item["runner_log"].splitlines() if line.strip()]
        print(f"--- {name} ({item['archive'].name}) ---")
        print("\n".join(lines[-8:]) if lines else "NO RUNNER LOG")


if __name__ == "__main__":
    main()
