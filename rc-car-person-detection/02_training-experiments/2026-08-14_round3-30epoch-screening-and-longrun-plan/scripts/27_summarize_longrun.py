from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    summaries: list[dict[str, object]] = []
    for state_path in sorted((ROOT / "results" / "longrun").glob("*/longrun_state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stage_states = list(state.get("stages", {}).values())
        if not stage_states:
            continue
        for stage_index, stage in enumerate(stage_states, 1):
            experiment = str(stage.get("experiment_name", ""))
            if not experiment:
                continue
            config_path = ROOT / "results" / "training" / experiment / "config.json"
            if not config_path.is_file():
                original_config = Path(str(stage.get("config", "")))
                config_path = original_config if original_config.is_file() else config_path
            if not config_path.is_file():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if (int(config["image_width"]), int(config["image_height"])) != (320, 240):
                continue
            history_path = ROOT / "results" / "training" / experiment / "history.csv"
            if not history_path.is_file():
                continue
            with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue
            best = max(rows, key=lambda row: number(row, "metric_map50_95", -1.0))
            last = max(rows, key=lambda row: number(row, "epoch", -1.0))
            summaries.append(
                {
                    "plan": f"{state.get('plan_name', state_path.parent.name)} / S{stage_index}",
                    "status": stage.get("status", state.get("status", "unknown")),
                    "experiment": experiment,
                    "last_epoch": int(number(last, "epoch")),
                    "best_epoch": int(number(best, "epoch")),
                    "map": number(best, "metric_map50_95"),
                    "ap75": number(best, "metric_ap75"),
                    "precision": number(best, "metric_precision"),
                    "recall": number(best, "metric_recall"),
                    "f1": number(best, "metric_f1"),
                    "tiny": number(best, "metric_tiny_lt16_recall"),
                    "small": number(best, "metric_small_16_32_recall"),
                    "warnings": sum(int(number(row, "warning_count")) for row in rows),
                }
            )
    if not summaries:
        print("No completed long-run final-stage history.csv files were found.")
        return 2
    summaries.sort(key=lambda item: float(item["map"]), reverse=True)
    print("=" * 164)
    print("LONG-RUN FINAL 320x240 VALID SUMMARY")
    print("=" * 164)
    print(
        f"{'PLAN':<42} {'STATUS':<10} {'LAST':>4} {'BEST':>4} {'mAP':>7} "
        f"{'AP75':>7} {'P':>7} {'R':>7} {'F1':>7} {'TINY':>7} {'SMALL':>7} {'WARN':>4}"
    )
    for row in summaries:
        print(
            f"{str(row['plan']):<42} {str(row['status']):<10} "
            f"{int(row['last_epoch']):>4} {int(row['best_epoch']):>4} "
            f"{float(row['map']):>7.4f} {float(row['ap75']):>7.4f} "
            f"{float(row['precision']):>7.4f} {float(row['recall']):>7.4f} "
            f"{float(row['f1']):>7.4f} {float(row['tiny']):>7.4f} "
            f"{float(row['small']):>7.4f} {int(row['warnings']):>4}"
        )
    print("=" * 164)
    print("Reference to beat: previous overall best mAP50:95 = 0.252478")
    print("Use only the final 320x240 stage for deployment selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
