from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

PARAM_COUNTS = {
    "school1_low_lr_100e": 345_018,
    "home_baseline_100e": 345_018,
    "school5_center_radius_100e": 345_018,
    "school2_lightweight_100e": 323_546,
    "school3_capacity_100e": 440_994,
    "school4_box_weight_100e": 345_018,
}


def linear_slope(frame: pd.DataFrame, column: str, count: int = 10) -> float:
    tail = frame[["epoch", column]].dropna().tail(count)
    if len(tail) < 2:
        return float("nan")
    return float(np.polyfit(tail["epoch"], tail[column], 1)[0])


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


experiments: list[dict[str, object]] = []
histories: dict[str, pd.DataFrame] = {}

history_files = sorted(
    ROOT.glob("results.*/results/training/*/history.csv"),
    key=lambda p: int(p.parts[-5].split(".")[-1]),
)
COMMON_EPOCH = min(int(pd.read_csv(path, usecols=["epoch"])["epoch"].max()) for path in history_files)

for archive_dir in sorted(ROOT.glob("results.*"), key=lambda p: int(p.name.split(".")[-1])):
    history_path = next(archive_dir.glob("results/training/*/history.csv"))
    experiment_dir = history_path.parent
    config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    device = json.loads((experiment_dir / "device.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(history_path).sort_values("epoch").reset_index(drop=True)
    name = str(config["experiment_name"])
    histories[name] = frame

    expected_epochs = int(config["epochs"])
    last_epoch = int(frame["epoch"].max())
    best_map_index = frame["metric_map50_95"].idxmax()
    best_map_row = frame.loc[best_map_index]
    best_f1_index = frame["metric_f1"].idxmax()
    best_f1_row = frame.loc[best_f1_index]
    best_tuned_f1_index = frame["metric_best_f1"].idxmax()
    best_tuned_f1_row = frame.loc[best_tuned_f1_index]
    min_valid_index = frame["valid_total"].idxmin()
    min_valid_row = frame.loc[min_valid_index]
    last_row = frame.iloc[-1]

    epoch_values = frame["epoch"].astype(int).tolist()
    contiguous = epoch_values == list(range(epoch_values[0], epoch_values[-1] + 1))
    duplicate_epochs = int(frame["epoch"].duplicated().sum())

    metric_columns = [
        col
        for col in frame.columns
        if col.startswith("train_") or col.startswith("valid_")
    ]
    nonfinite_counts: dict[str, int] = {}
    for column in metric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        count = int((~np.isfinite(numeric)).sum())
        if count:
            nonfinite_counts[column] = count

    warning_path = experiment_dir / "warnings.log"
    warning_text = warning_path.read_text(encoding="utf-8") if warning_path.exists() else ""
    warning_epochs = sorted({int(x) for x in re.findall(r"epoch (\d+):", warning_text)})

    common_row = frame.loc[frame["epoch"] == COMMON_EPOCH].iloc[0]
    up_to_common = frame.loc[frame["epoch"] <= COMMON_EPOCH]
    best_common_row = up_to_common.loc[up_to_common["metric_map50_95"].idxmax()]
    config_mtime = datetime.fromtimestamp((experiment_dir / "config.json").stat().st_mtime)
    history_mtime = datetime.fromtimestamp(history_path.stat().st_mtime)
    best_mtime = datetime.fromtimestamp((experiment_dir / "checkpoints" / "best.pt").stat().st_mtime)
    last_mtime = datetime.fromtimestamp((experiment_dir / "checkpoints" / "last.pt").stat().st_mtime)
    recorded_hours = float(frame["seconds"].sum() / 3600.0)
    wall_hours = (history_mtime - config_mtime).total_seconds() / 3600.0

    experiments.append(
        {
            "archive": archive_dir.name,
            "experiment": name,
            "setting_learning_rate": float(config["learning_rate"]),
            "setting_fpn_channels": int(config["fpn_channels"]),
            "setting_backbone_expansion": float(config["backbone_expansion"]),
            "setting_box_loss_weight": float(config["box_loss_weight"]),
            "setting_center_radius": float(config["center_sampling_radius"]),
            "parameters": PARAM_COUNTS[name],
            "torch_version": device["torch_version"],
            "started_at": config_mtime.isoformat(sep=" ", timespec="seconds"),
            "last_completed_at": history_mtime.isoformat(sep=" ", timespec="seconds"),
            "best_checkpoint_at": best_mtime.isoformat(sep=" ", timespec="seconds"),
            "last_checkpoint_at": last_mtime.isoformat(sep=" ", timespec="seconds"),
            "last_epoch": last_epoch,
            "completion_percent": last_epoch / expected_epochs * 100.0,
            "history_rows": len(frame),
            "contiguous_epochs": contiguous,
            "duplicate_epochs": duplicate_epochs,
            "best_map50_95": float(best_map_row["metric_map50_95"]),
            "best_map_epoch": int(best_map_row["epoch"]),
            "best_map_ap50": float(best_map_row["metric_ap50"]),
            "best_map_ap75": float(best_map_row["metric_ap75"]),
            "best_map_precision_at_025": float(best_map_row["metric_precision"]),
            "best_map_recall_at_025": float(best_map_row["metric_recall"]),
            "best_map_f1_at_025": float(best_map_row["metric_f1"]),
            "best_map_threshold_tuned_f1": float(best_map_row["metric_best_f1"]),
            "best_map_tuned_threshold": float(best_map_row["metric_best_f1_threshold"]),
            "best_map_tiny_recall": float(best_map_row["metric_tiny_lt16_recall"]),
            "best_map_small_recall": float(best_map_row["metric_small_16_32_recall"]),
            "best_map_medium_recall": float(best_map_row["metric_medium_32_96_recall"]),
            "best_map_large_recall": float(best_map_row["metric_large_ge96_recall"]),
            "best_fixed_f1": float(best_f1_row["metric_f1"]),
            "best_fixed_f1_epoch": int(best_f1_row["epoch"]),
            "best_threshold_tuned_f1": float(best_tuned_f1_row["metric_best_f1"]),
            "best_threshold_tuned_f1_epoch": int(best_tuned_f1_row["epoch"]),
            "best_threshold": float(best_tuned_f1_row["metric_best_f1_threshold"]),
            "min_valid_loss": float(min_valid_row["valid_total"]),
            "min_valid_loss_epoch": int(min_valid_row["epoch"]),
            "last_train_loss": float(last_row["train_total"]),
            "last_valid_loss": float(last_row["valid_total"]),
            "last_map50_95": float(last_row["metric_map50_95"]),
            "last_f1_at_025": float(last_row["metric_f1"]),
            "map_slope_last10": linear_slope(frame, "metric_map50_95"),
            "train_loss_slope_last10": linear_slope(frame, "train_total"),
            "valid_loss_slope_last10": linear_slope(frame, "valid_total"),
            "mean_epoch_seconds": float(frame["seconds"].mean()),
            "total_recorded_hours": recorded_hours,
            "wall_elapsed_hours": wall_hours,
            "unaccounted_minutes": (wall_hours - recorded_hours) * 60.0,
            "mean_train_images_per_second": float(frame["train_images_per_second"].mean()),
            "peak_vram_gib": float(frame["peak_vram_gib"].max()),
            "warning_epochs": len(warning_epochs),
            "warning_epoch_percent": len(warning_epochs) / len(frame) * 100.0,
            "warning_epoch_list": ",".join(map(str, warning_epochs)),
            "nonfinite_columns": json.dumps(nonfinite_counts, ensure_ascii=False),
            "final_amp_scale": finite_float(last_row["amp_scale"]),
            "common_epoch": COMMON_EPOCH,
            "common_map50_95": float(common_row["metric_map50_95"]),
            "common_precision": float(common_row["metric_precision"]),
            "common_recall": float(common_row["metric_recall"]),
            "common_f1": float(common_row["metric_f1"]),
            "common_tiny_recall": float(common_row["metric_tiny_lt16_recall"]),
            "common_small_recall": float(common_row["metric_small_16_32_recall"]),
            "best_map_through_common": float(best_common_row["metric_map50_95"]),
            "best_map_through_common_epoch": int(best_common_row["epoch"]),
        }
    )


summary = pd.DataFrame(experiments).sort_values("best_map50_95", ascending=False)
summary.to_csv(ROOT / "experiment_summary.csv", index=False, encoding="utf-8-sig")

common_columns = [
    "archive",
    "experiment",
    "parameters",
    "common_epoch",
    "common_map50_95",
    "common_precision",
    "common_recall",
    "common_f1",
    "common_tiny_recall",
    "common_small_recall",
    "best_map_through_common",
    "best_map_through_common_epoch",
]
summary[common_columns].sort_values("common_map50_95", ascending=False).to_csv(
    ROOT / f"common_epoch{COMMON_EPOCH}_comparison.csv", index=False, encoding="utf-8-sig"
)

long_frames = []
for name, frame in histories.items():
    copy = frame.copy()
    copy.insert(0, "experiment", name)
    long_frames.append(copy)
pd.concat(long_frames, ignore_index=True).to_csv(
    ROOT / "all_histories.csv", index=False, encoding="utf-8-sig"
)

display_columns = [
    "archive",
    "experiment",
    "last_epoch",
    "best_map50_95",
    "best_map_epoch",
    "best_map_precision_at_025",
    "best_map_recall_at_025",
    "best_map_f1_at_025",
    "best_map_tiny_recall",
    "best_map_small_recall",
    "best_fixed_f1",
    "best_fixed_f1_epoch",
    "min_valid_loss",
    "min_valid_loss_epoch",
    "warning_epochs",
    "mean_epoch_seconds",
]
print("\nOVERALL\n")
print(summary[display_columns].to_string(index=False))
print(f"\nCOMMON EPOCH {COMMON_EPOCH}\n")
print(
    summary[common_columns]
    .sort_values("common_map50_95", ascending=False)
    .to_string(index=False)
)
print("\nSTOP / HEALTH\n")
print(
    summary[
        [
            "archive",
            "experiment",
            "last_epoch",
            "last_map50_95",
            "last_train_loss",
            "last_valid_loss",
            "map_slope_last10",
            "valid_loss_slope_last10",
            "warning_epoch_percent",
            "nonfinite_columns",
            "total_recorded_hours",
            "mean_train_images_per_second",
            "peak_vram_gib",
            "torch_version",
        ]
    ].to_string(index=False)
)
