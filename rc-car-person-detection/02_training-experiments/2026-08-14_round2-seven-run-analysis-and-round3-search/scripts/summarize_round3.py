from __future__ import annotations

import csv
from pathlib import Path


PREFIX = "r3_school"


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    root = Path(__file__).resolve().parent
    training_root = root / "results" / "training"
    summaries = []
    for experiment_dir in sorted(training_root.glob(f"{PREFIX}*")):
        history = experiment_dir / "history.csv"
        if not history.is_file():
            continue
        with history.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        best = max(rows, key=lambda row: number(row, "metric_map50_95", -1.0))
        last = max(rows, key=lambda row: number(row, "epoch", -1.0))
        summaries.append(
            {
                "experiment": experiment_dir.name,
                "last_epoch": int(number(last, "epoch")),
                "best_epoch": int(number(best, "epoch")),
                "map50_95": number(best, "metric_map50_95"),
                "ap50": number(best, "metric_ap50"),
                "ap75": number(best, "metric_ap75"),
                "precision": number(best, "metric_precision"),
                "recall": number(best, "metric_recall"),
                "f1": number(best, "metric_f1"),
                "tiny_recall": number(best, "metric_tiny_lt16_recall"),
                "small_recall": number(best, "metric_small_16_32_recall"),
                "warnings": sum(int(number(row, "warning_count")) for row in rows),
            }
        )

    if not summaries:
        print("No Round 3 history.csv found yet.")
        return 2
    summaries.sort(key=lambda row: row["map50_95"], reverse=True)
    print("=" * 132)
    print("ROUND 3 CURRENT BEST SUMMARY - VALID SET ONLY")
    print("=" * 132)
    print(
        f"{'EXPERIMENT':<40} {'LAST':>4} {'BEST':>4} {'mAP':>7} {'AP50':>7} {'AP75':>7} "
        f"{'P':>7} {'R':>7} {'F1':>7} {'TINY':>7} {'SMALL':>7} {'WARN':>4}"
    )
    for row in summaries:
        print(
            f"{row['experiment']:<40} {row['last_epoch']:>4} {row['best_epoch']:>4} "
            f"{row['map50_95']:>7.4f} {row['ap50']:>7.4f} {row['ap75']:>7.4f} "
            f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f} "
            f"{row['tiny_recall']:>7.4f} {row['small_recall']:>7.4f} {row['warnings']:>4}"
        )
    print("=" * 132)
    print("Reference to beat: Round 1 results.4 mAP50:95 = 0.252478")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
