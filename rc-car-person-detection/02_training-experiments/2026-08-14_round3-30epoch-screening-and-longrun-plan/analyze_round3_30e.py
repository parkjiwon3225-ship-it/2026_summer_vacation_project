"""Recompute the Round 3 ranking from the published combined history file.

The raw result archives and checkpoints are intentionally not duplicated in GitHub.
This script therefore uses ``round3_30e_all_histories.csv`` and writes a compact,
reproducible comparison file beside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "round3_30e_all_histories.csv"
OUTPUT = ROOT / "round3_30e_recomputed_summary.csv"
REFERENCE_MAP = 0.2524780012739939  # previous results.4 validation best

CONFIG_BY_RESULT = {
    "results.14": {"seed": 11, "expansion": 2.0, "box_weight": 2.0, "radius": 1.5},
    "results.15": {"seed": 14, "expansion": 2.0, "box_weight": 2.0, "radius": 1.5},
    "results.16": {"seed": 11, "expansion": 2.5, "box_weight": 2.0, "radius": 1.5},
    "results.17": {"seed": 14, "expansion": 2.5, "box_weight": 2.0, "radius": 1.5},
    "results.18": {"seed": 11, "expansion": 2.0, "box_weight": 2.5, "radius": 1.5},
    "results.19": {"seed": 11, "expansion": 2.0, "box_weight": 2.0, "radius": 2.0},
}


def finite_slope(values: pd.Series) -> float:
    data = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        return float("nan")
    x = np.arange(len(data), dtype=np.float64)
    return float(np.polyfit(x, data.to_numpy(dtype=np.float64), 1)[0])


def main() -> None:
    history = pd.read_csv(INPUT)
    required = {
        "result",
        "experiment",
        "epoch",
        "train_total",
        "valid_total",
        "metric_map50_95",
        "metric_ap50",
        "metric_ap75",
        "metric_precision",
        "metric_recall",
        "metric_f1",
        "metric_tiny_lt16_recall",
        "metric_small_16_32_recall",
        "warning_count",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for (result, experiment), run in history.groupby(["result", "experiment"], sort=False):
        run = run.sort_values("epoch").reset_index(drop=True)
        best = run.loc[int(run["metric_map50_95"].idxmax())]
        last = run.iloc[-1]
        common = run[run["epoch"] <= 28]
        common_late = common[common["epoch"] >= 20]
        best_common = common.loc[int(common["metric_map50_95"].idxmax())]
        last5 = run.tail(min(5, len(run)))
        last10 = run.tail(min(10, len(run)))
        config = CONFIG_BY_RESULT[str(result)]

        rows.append(
            {
                "result": result,
                "experiment": experiment,
                "seed": config["seed"],
                "fpn_channels": 48,
                "backbone_expansion": config["expansion"],
                "box_loss_weight": config["box_weight"],
                "center_sampling_radius": config["radius"],
                "epochs_completed": int(last["epoch"]),
                "best_epoch": int(best["epoch"]),
                "best_map50_95": float(best["metric_map50_95"]),
                "delta_vs_results4": float(best["metric_map50_95"] - REFERENCE_MAP),
                "best_ap50": float(best["metric_ap50"]),
                "best_ap75": float(best["metric_ap75"]),
                "best_precision": float(best["metric_precision"]),
                "best_recall": float(best["metric_recall"]),
                "best_f1": float(best["metric_f1"]),
                "best_tiny_recall": float(best["metric_tiny_lt16_recall"]),
                "best_small_recall": float(best["metric_small_16_32_recall"]),
                "common28_best_epoch": int(best_common["epoch"]),
                "common28_best_map50_95": float(best_common["metric_map50_95"]),
                "common20_28_map_mean": float(common_late["metric_map50_95"].mean()),
                "last5_map_mean": float(last5["metric_map50_95"].mean()),
                "last5_map_std": float(last5["metric_map50_95"].std(ddof=0)),
                "last10_map_slope": finite_slope(last10["metric_map50_95"]),
                "last10_valid_loss_slope": finite_slope(last10["valid_total"]),
                "last_train_total": float(last["train_total"]),
                "last_valid_total": float(last["valid_total"]),
                "warning_count_total": int(
                    pd.to_numeric(run["warning_count"], errors="coerce").fillna(0).sum()
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values("best_map50_95", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    columns = [
        "rank",
        "result",
        "experiment",
        "epochs_completed",
        "best_epoch",
        "best_map50_95",
        "best_ap50",
        "best_ap75",
        "best_precision",
        "best_recall",
        "best_f1",
        "best_tiny_recall",
        "best_small_recall",
        "last5_map_mean",
        "warning_count_total",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary[columns].to_string(index=False))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
