from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]

METRICS = {
    "map50_95": "metric_map50_95",
    "ap50": "metric_ap50",
    "ap75": "metric_ap75",
    "precision": "metric_precision",
    "recall": "metric_recall",
    "f1": "metric_f1",
    "best_f1": "metric_best_f1",
    "best_f1_threshold": "metric_best_f1_threshold",
    "tiny_recall": "metric_tiny_lt16_recall",
    "small_recall": "metric_small_16_32_recall",
    "regular_recall": "metric_regular_ge32_recall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze copied RC person-detector experiment folders."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Folder containing experiment folders. Default: ROOT/results/training",
    )
    parser.add_argument("--expected", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder. Default: ROOT/results/round1_analysis",
    )
    return parser.parse_args()


def number(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def finite(value: Any) -> bool:
    return math.isfinite(number(value))


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_at_max(rows: list[dict[str, str]], column: str) -> dict[str, str] | None:
    candidates = [row for row in rows if finite(row.get(column))]
    return max(candidates, key=lambda row: number(row[column])) if candidates else None


def row_for_epoch(rows: list[dict[str, str]], epoch: int) -> dict[str, str] | None:
    eligible = [row for row in rows if int(number(row.get("epoch"), -1)) <= epoch]
    return max(eligible, key=lambda row: int(number(row.get("epoch"), -1))) if eligible else None


def warning_summary(path: Path) -> tuple[int, str]:
    if not path.is_file():
        return 0, ""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    messages = []
    for line in lines:
        messages.append(line.split(":", 1)[1].strip() if ":" in line else line)
    counts = Counter(messages)
    summary = "; ".join(f"{message} x{count}" for message, count in counts.most_common())
    return len(lines), summary


def metric_values(row: dict[str, str] | None, prefix: str = "") -> dict[str, Any]:
    if row is None:
        return {f"{prefix}{name}": "" for name in METRICS}
    return {
        f"{prefix}{name}": number(row.get(column)) if finite(row.get(column)) else ""
        for name, column in METRICS.items()
    }


def selection_score(record: dict[str, Any]) -> float:
    # Sorting aid only. Final selection must inspect every metric and learning curve.
    weights = {
        "best_map50_95": 0.30,
        "best_recall": 0.20,
        "best_f1": 0.15,
        "best_tiny_recall": 0.20,
        "best_small_recall": 0.15,
    }
    values = []
    total_weight = 0.0
    for key, weight in weights.items():
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value) * weight)
            total_weight += weight
    return sum(values) / total_weight if total_weight else float("-inf")


def inspect_experiment(experiment_dir: Path) -> dict[str, Any]:
    history_path = experiment_dir / "history.csv"
    config_path = experiment_dir / "config.json"
    device_path = experiment_dir / "device.json"
    warnings_path = experiment_dir / "warnings.log"
    best_path = experiment_dir / "checkpoints" / "best.pt"
    last_path = experiment_dir / "checkpoints" / "last.pt"

    rows = read_history(history_path)
    if not rows:
        raise ValueError("history.csv has no data rows")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    device = json.loads(device_path.read_text(encoding="utf-8")) if device_path.is_file() else {}
    rows.sort(key=lambda row: int(number(row.get("epoch"), -1)))
    last_row = rows[-1]
    best_row = row_at_max(rows, METRICS["map50_95"])
    warning_count, warning_types = warning_summary(warnings_path)

    last_epoch = int(number(last_row.get("epoch"), 0))
    configured_epochs = int(number(config.get("epochs"), last_epoch))
    recent_seconds = [number(row.get("seconds")) for row in rows[-5:] if finite(row.get("seconds"))]
    recorded_speeds = [
        number(row.get("train_images_per_second"))
        for row in rows
        if finite(row.get("train_images_per_second"))
    ]
    nonfinite_rows = sum(
        not all(finite(row.get(key)) for key in ("train_total", "valid_total"))
        for row in rows
    )
    status_issues = []
    if not config_path.is_file():
        status_issues.append("missing config.json")
    if not best_path.is_file():
        status_issues.append("missing best.pt")
    if not last_path.is_file():
        status_issues.append("missing last.pt")
    if nonfinite_rows:
        status_issues.append(f"non-finite loss rows={nonfinite_rows}")
    if last_epoch < configured_epochs:
        status_issues.append(f"incomplete {last_epoch}/{configured_epochs}")

    result: dict[str, Any] = {
        "run_id": experiment_dir.name,
        "experiment": str(config.get("experiment_name", experiment_dir.name)),
        "folder": str(experiment_dir),
        "status": "PASS" if not status_issues else "CHECK",
        "issues": "; ".join(status_issues),
        "last_epoch": last_epoch,
        "configured_epochs": configured_epochs,
        "completion_percent": 100.0 * last_epoch / max(configured_epochs, 1),
        "best_epoch": int(number(best_row.get("epoch"), 0)) if best_row else "",
        "best_pt": best_path.is_file(),
        "last_pt": last_path.is_file(),
        "warning_lines": warning_count,
        "warning_types": warning_types,
        "nonfinite_loss_rows": nonfinite_rows,
        "learning_rate": number(config.get("learning_rate")) if config else "",
        "image_width": config.get("image_width", ""),
        "image_height": config.get("image_height", ""),
        "batch_size": config.get("batch_size", ""),
        "fpn_channels": config.get("fpn_channels", ""),
        "backbone_expansion": config.get("backbone_expansion", ""),
        "box_loss_weight": config.get("box_loss_weight", ""),
        "center_sampling_radius": config.get("center_sampling_radius", ""),
        "gpu_name": device.get("gpu_name", ""),
        "best_train_total": number(best_row.get("train_total")) if best_row and finite(best_row.get("train_total")) else "",
        "best_valid_total": number(best_row.get("valid_total")) if best_row and finite(best_row.get("valid_total")) else "",
        "last_train_total": number(last_row.get("train_total")) if finite(last_row.get("train_total")) else "",
        "last_valid_total": number(last_row.get("valid_total")) if finite(last_row.get("valid_total")) else "",
        "last_map50_95": number(last_row.get(METRICS["map50_95"])) if finite(last_row.get(METRICS["map50_95"])) else "",
        "mean_last5_epoch_seconds": mean(recent_seconds) if recent_seconds else "",
        "total_recorded_hours": sum(
            number(row.get("seconds")) for row in rows if finite(row.get("seconds"))
        ) / 3600.0,
        "peak_vram_gib": max(
            [number(row.get("peak_vram_gib")) for row in rows if finite(row.get("peak_vram_gib"))],
            default=float("nan"),
        ),
        "mean_images_per_second": mean(recorded_speeds) if recorded_speeds else "",
    }
    result.update(metric_values(best_row, "best_"))
    result["selection_score"] = selection_score(result)
    return result


def format_value(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_report(
    path: Path,
    records: list[dict[str, Any]],
    common_epoch: int,
    expected: int,
    results_root: Path,
) -> None:
    ranking = sorted(records, key=lambda record: record["selection_score"], reverse=True)
    lines = [
        "# Round 1 Experiment Analysis",
        "",
        f"- Results root: `{results_root}`",
        f"- Experiments found: **{len(records)}** / expected **{expected}**",
        f"- Common comparison epoch: **{common_epoch}**",
        "- Test split was not used.",
        "",
        "> `selection_score` is only a sorting aid: 30% mAP50:95, 20% Recall, "
        "15% F1, 20% tiny Recall, 15% small Recall. Final selection must inspect "
        "learning curves, warnings, precision and model size.",
        "",
        "## Best-mAP checkpoint comparison",
        "",
        "|Rank|Experiment|Status|Epoch|mAP|Precision|Recall|F1|Tiny R|Small R|Valid loss|Warnings|Score|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(ranking, start=1):
        lines.append(
            "|{rank}|{name}|{status}|{best}/{last}|{map}|{precision}|{recall}|{f1}|"
            "{tiny}|{small}|{valid}|{warnings}|{score}|".format(
                rank=rank,
                name=record["run_id"],
                status=record["status"],
                best=record["best_epoch"],
                last=record["last_epoch"],
                map=format_value(record["best_map50_95"]),
                precision=format_value(record["best_precision"]),
                recall=format_value(record["best_recall"]),
                f1=format_value(record["best_f1"]),
                tiny=format_value(record["best_tiny_recall"]),
                small=format_value(record["best_small_recall"]),
                valid=format_value(record["best_valid_total"]),
                warnings=record["warning_lines"],
                score=format_value(record["selection_score"]),
            )
        )
    lines.extend(["", "## Integrity and completion", ""])
    for record in records:
        detail = record["issues"] or "complete; required files present"
        lines.append(
            f"- **{record['run_id']}**: {record['status']} - {detail}; "
            f"warnings={record['warning_lines']}"
        )
    lines.extend(
        [
            "",
            "## Decision checklist for Round 2",
            "",
            "1. Exclude or repair experiments with missing checkpoints or non-finite losses.",
            "2. Compare all experiments at the common epoch in `comparison_common_epoch.csv` if any run is incomplete.",
            "3. Select the best structure using mAP, Recall, F1 and tiny/small Recall together.",
            "4. Treat a higher Recall with a large Precision collapse as a trade-off, not an automatic win.",
            "5. Use Round 2 to repeat the winner with a new seed and combine only variables that showed a benefit.",
            "6. Do not evaluate the test split until the final 1-2 candidates are frozen.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_root = (args.results_root or root / "results" / "training").resolve()
    output = (args.output or root / "results" / "round1_analysis").resolve()
    history_paths = sorted(results_root.rglob("history.csv"))
    if not history_paths:
        raise FileNotFoundError(f"No history.csv found under {results_root}")

    records = []
    errors = []
    for history_path in history_paths:
        try:
            records.append(inspect_experiment(history_path.parent))
        except Exception as error:
            errors.append(f"{history_path.parent}: {error}")
    if not records:
        raise RuntimeError("Every experiment failed inspection:\n" + "\n".join(errors))

    common_epoch = min(int(record["last_epoch"]) for record in records)
    common_records = []
    for record in records:
        rows = read_history(Path(record["folder"]) / "history.csv")
        row = row_for_epoch(rows, common_epoch)
        common = {
            "run_id": record["run_id"],
            "experiment": record["experiment"],
            "common_epoch": common_epoch,
            "train_total": number(row.get("train_total")) if row and finite(row.get("train_total")) else "",
            "valid_total": number(row.get("valid_total")) if row and finite(row.get("valid_total")) else "",
        }
        common.update(metric_values(row))
        common_records.append(common)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison_best_map.csv", records)
    write_csv(output / "comparison_common_epoch.csv", common_records)
    write_report(output / "ROUND1_ANALYSIS.md", records, common_epoch, args.expected, results_root)
    (output / "analysis.json").write_text(
        json.dumps(
            {
                "results_root": str(results_root),
                "expected": args.expected,
                "found": len(records),
                "common_epoch": common_epoch,
                "inspection_errors": errors,
                "experiments": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 92)
    print("ROUND 1 EXPERIMENT ANALYSIS")
    print("=" * 92)
    print(f"Results found : {len(records)} / expected {args.expected}")
    print(f"Common epoch  : {common_epoch}")
    print(f"Output        : {output}")
    print("-" * 92)
    ranking = sorted(records, key=lambda record: record["selection_score"], reverse=True)
    print(f"{'RANK':>4} {'EXPERIMENT':<31} {'MAP':>7} {'REC':>7} {'F1':>7} {'TINY':>7} {'SMALL':>7} {'STATUS':>8}")
    for rank, record in enumerate(ranking, start=1):
        print(
            f"{rank:>4} {record['run_id']:<31} "
            f"{format_value(record['best_map50_95']):>7} "
            f"{format_value(record['best_recall']):>7} "
            f"{format_value(record['best_f1']):>7} "
            f"{format_value(record['best_tiny_recall']):>7} "
            f"{format_value(record['best_small_recall']):>7} "
            f"{record['status']:>8}"
        )
    if errors:
        print("-" * 92)
        for error in errors:
            print(f"INSPECTION ERROR: {error}")
    if len(records) != args.expected or errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
