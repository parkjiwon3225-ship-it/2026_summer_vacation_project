from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def deduplicated_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_epoch: dict[int, dict[str, str]] = {}
    for row in rows:
        by_epoch[int(number(row, "epoch", -1))] = row
    return [by_epoch[key] for key in sorted(by_epoch) if key >= 0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize home final search progress.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--plan", type=Path, default=Path("plans/home_final_search.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    plan_path = args.plan.resolve() if args.plan.is_absolute() else root / args.plan
    plan = load_json(plan_path)
    state_path = (
        root
        / "results"
        / "home_final_search"
        / str(plan["plan_name"])
        / "search_state.json"
    )
    state = load_json(state_path) if state_path.is_file() else {}

    summaries: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(plan["stages"], 1):
        config_path = root / str(stage["config"])
        config = load_json(config_path)
        experiment = str(config["experiment_name"])
        experiment_dir = root / "results" / "training" / experiment
        history_path = experiment_dir / "history.csv"
        if not history_path.is_file():
            continue
        rows = deduplicated_history(history_path)
        if not rows:
            continue
        best = max(
            rows,
            key=lambda row: (
                number(row, "metric_map50_95", -1.0),
                number(row, "metric_small_16_32_recall", -1.0),
                number(row, "metric_recall", -1.0),
                number(row, "metric_ap75", -1.0),
            ),
        )
        last = rows[-1]
        stage_state = state.get("stages", {}).get(str(stage["name"]), {})
        summaries.append(
            {
                "stage": stage_index,
                "stage_name": stage["name"],
                "status": stage_state.get("status", "history_only"),
                "experiment": experiment,
                "last_epoch": int(number(last, "epoch", -1)),
                "best_epoch": int(number(best, "epoch", -1)),
                "map50_95": number(best, "metric_map50_95", -1.0),
                "ap50": number(best, "metric_ap50", -1.0),
                "ap75": number(best, "metric_ap75", -1.0),
                "precision": number(best, "metric_precision", -1.0),
                "recall": number(best, "metric_recall", -1.0),
                "f1": number(best, "metric_f1", -1.0),
                "best_f1": number(best, "metric_best_f1", -1.0),
                "best_f1_threshold": number(best, "metric_best_f1_threshold", -1.0),
                "tiny_recall": number(best, "metric_tiny_lt16_recall", -1.0),
                "small_recall": number(best, "metric_small_16_32_recall", -1.0),
                "valid_total": number(best, "valid_total", -1.0),
                "warning_count": sum(int(number(row, "warning_count")) for row in rows),
            }
        )

    if not summaries:
        print("No home final-search history.csv files were found.")
        return 2
    summaries.sort(
        key=lambda item: (
            float(item["map50_95"]),
            float(item["small_recall"]),
            float(item["recall"]),
            float(item["ap75"]),
        ),
        reverse=True,
    )
    for rank, item in enumerate(summaries, 1):
        item["rank"] = rank

    output_dir = root / "results" / "home_final_search" / str(plan["plan_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "home_final_ranking.csv"
    fieldnames = ["rank"] + [key for key in summaries[0] if key != "rank"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print("=" * 160)
    print("HOME FINAL SEARCH VALID RANKING")
    print("=" * 160)
    print(
        f"{'RANK':>4} {'STATUS':<14} {'LAST':>4} {'BEST':>4} {'mAP':>7} "
        f"{'AP75':>7} {'P':>7} {'R':>7} {'F1':>7} {'TINY':>7} {'SMALL':>7} {'WARN':>4}  EXPERIMENT"
    )
    for item in summaries:
        print(
            f"{int(item['rank']):>4} {str(item['status']):<14} "
            f"{int(item['last_epoch']):>4} {int(item['best_epoch']):>4} "
            f"{float(item['map50_95']):>7.4f} {float(item['ap75']):>7.4f} "
            f"{float(item['precision']):>7.4f} {float(item['recall']):>7.4f} "
            f"{float(item['f1']):>7.4f} {float(item['tiny_recall']):>7.4f} "
            f"{float(item['small_recall']):>7.4f} {int(item['warning_count']):>4}  "
            f"{item['experiment']}"
        )
    print("=" * 160)
    print(f"Reference results.14 mAP50:95 : {float(plan['reference_map50_95']):.6f}")
    print(f"Current best                    : {float(summaries[0]['map50_95']):.6f}")
    print(f"Ranking CSV                    : {output_path}")
    print("Selection split                : Valid only; Test not used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
