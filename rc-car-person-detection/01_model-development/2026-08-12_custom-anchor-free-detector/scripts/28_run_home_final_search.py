from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # A process interruption between CSV append and checkpoint replacement can
    # repeat one epoch on resume. Keep the last record for each epoch.
    by_epoch: dict[int, dict[str, str]] = {}
    for row in rows:
        by_epoch[int(number(row, "epoch", -1))] = row
    return [by_epoch[key] for key in sorted(by_epoch) if key >= 0]


def checkpoint_epoch_and_config(path: Path) -> tuple[int, dict[str, Any]]:
    # Torch is imported only for an actual run, so package-only dry-run and
    # structural verification do not require a local Torch installation.
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise TypeError(f"Unsupported resume checkpoint: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"Checkpoint has no config object: {path}")
    return int(checkpoint["epoch"]), config


def validate_resume_compatibility(path: Path, current: dict[str, Any]) -> int:
    checkpoint_epoch, saved = checkpoint_epoch_and_config(path)
    keys = (
        "seed",
        "image_width",
        "image_height",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "fpn_channels",
        "backbone_expansion",
        "box_loss_weight",
        "quality_loss_weight",
        "center_sampling_radius",
    )
    mismatches = [
        f"{key}: checkpoint={saved.get(key)!r}, current={current.get(key)!r}"
        for key in keys
        if saved.get(key) != current.get(key)
    ]
    if mismatches:
        raise ValueError("Resume config mismatch:\n" + "\n".join(mismatches))
    if checkpoint_epoch > int(current["epochs"]):
        raise ValueError(
            f"Checkpoint epoch {checkpoint_epoch} exceeds requested epochs {current['epochs']}"
        )
    return checkpoint_epoch


def sanitize_history_for_resume(path: Path, checkpoint_epoch: int) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        original = list(csv.DictReader(stream))
    if not original:
        return
    by_epoch: dict[int, dict[str, str]] = {}
    for row in original:
        epoch = int(number(row, "epoch", -1))
        if 0 <= epoch <= checkpoint_epoch:
            by_epoch[epoch] = row
    cleaned = [by_epoch[key] for key in sorted(by_epoch)]
    if len(cleaned) == len(original) and all(
        int(number(row, "epoch", -1)) == key
        for row, key in zip(original, sorted(by_epoch))
    ):
        return
    backup = path.with_name(path.stem + ".pre_resume_backup.csv")
    if not backup.exists():
        shutil.copy2(path, backup)
    fieldnames = list(original[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)
    temporary.replace(path)


def best_record(root: Path, experiment_name: str) -> dict[str, Any] | None:
    experiment = root / "results" / "training" / experiment_name
    history_path = experiment / "history.csv"
    checkpoint = experiment / "checkpoints" / "best.pt"
    if not history_path.is_file() or not checkpoint.is_file():
        return None
    rows = read_history(history_path)
    if not rows:
        return None
    best = max(
        rows,
        key=lambda row: (
            number(row, "metric_map50_95", -1.0),
            number(row, "metric_small_16_32_recall", -1.0),
            number(row, "metric_recall", -1.0),
            number(row, "metric_ap75", -1.0),
        ),
    )
    return {
        "experiment": experiment_name,
        "checkpoint": checkpoint,
        "history": history_path,
        "best_row": best,
        "map50_95": number(best, "metric_map50_95", -1.0),
        "small_recall": number(best, "metric_small_16_32_recall", -1.0),
        "recall": number(best, "metric_recall", -1.0),
        "ap75": number(best, "metric_ap75", -1.0),
        "epoch": int(number(best, "epoch", -1)),
    }


def choose_global_best(root: Path, experiment_names: list[str]) -> dict[str, Any]:
    candidates = [best_record(root, name) for name in experiment_names]
    available = [candidate for candidate in candidates if candidate is not None]
    if not available:
        raise FileNotFoundError("No prior best.pt and history.csv pair is available")
    return max(
        available,
        key=lambda item: (
            float(item["map50_95"]),
            float(item["small_recall"]),
            float(item["recall"]),
            float(item["ap75"]),
        ),
    )


def validate_plan(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    required = {"plan_name", "total_budget_hours", "seed_directory", "stages"}
    missing = sorted(required.difference(plan))
    if missing:
        raise KeyError(f"Plan is missing keys: {missing}")
    stages = plan["stages"]
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("Home final search must contain exactly three stages")
    budget_sum = sum(float(stage["budget_hours"]) for stage in stages)
    if abs(budget_sum - float(plan["total_budget_hours"])) > 1e-6:
        raise ValueError("Stage budgets must equal total_budget_hours")

    normalized: list[dict[str, Any]] = []
    experiments: set[str] = set()
    for index, stage in enumerate(stages):
        config_path = resolve_from_root(root, str(stage["config"]))
        config = load_json(config_path)
        experiment = str(config["experiment_name"])
        if experiment in experiments:
            raise ValueError(f"Duplicate experiment_name: {experiment}")
        experiments.add(experiment)
        expected_mode = "resume_seeded" if index == 0 else "init_global_best"
        if stage.get("mode") != expected_mode:
            raise ValueError(f"Stage {index + 1} mode must be {expected_mode}")
        if (int(config["image_width"]), int(config["image_height"])) != (320, 240):
            raise ValueError(f"All home stages must be 320x240: {config_path}")
        expected = {
            "fpn_channels": 48,
            "backbone_expansion": 2.0,
            "box_loss_weight": 2.0,
            "center_sampling_radius": 1.5,
            "checkpoint_every": 1,
            "metrics_every": 1,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"{config_path.name}: {key} must be {value}")
        normalized.append(
            {
                **stage,
                "config_path": config_path,
                "config": config,
                "experiment": experiment,
                "budget_hours": float(stage["budget_hours"]),
            }
        )

    seed = resolve_from_root(root, str(plan["seed_directory"]))
    for required_file in (
        seed / "checkpoints" / "best.pt",
        seed / "checkpoints" / "last.pt",
        seed / "history.csv",
        seed / "config.json",
        seed / "device.json",
    ):
        if not required_file.is_file():
            raise FileNotFoundError(f"Seed file is missing: {required_file}")
    seed_rows = read_history(seed / "history.csv")
    seed_best = max(number(row, "metric_map50_95", -1.0) for row in seed_rows)
    reference = float(plan.get("reference_map50_95", seed_best))
    if abs(seed_best - reference) > 1e-6:
        raise ValueError(f"Seed mAP mismatch: history={seed_best} plan={reference}")
    return normalized


def seed_first_stage(root: Path, seed: Path, experiment_name: str) -> None:
    target = root / "results" / "training" / experiment_name
    checkpoint_dir = target / "checkpoints"
    target_last = checkpoint_dir / "last.pt"
    if target_last.is_file():
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(
            f"Stage-1 folder exists without last.pt; preserve it for diagnosis: {target}"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed / "checkpoints" / "best.pt", checkpoint_dir / "best.pt")
    shutil.copy2(seed / "checkpoints" / "last.pt", checkpoint_dir / "last.pt")
    shutil.copy2(seed / "history.csv", target / "history.csv")
    shutil.copy2(seed / "config.json", target / "source_config.json")
    shutil.copy2(seed / "device.json", target / "source_device.json")
    if (seed / "runner_console.log").is_file():
        shutil.copy2(seed / "runner_console.log", target / "source_runner_console.log")
    provenance = {
        "source": "results.14.zip / r3_school1_fpn48_exp200_seed11",
        "source_best_epoch": 29,
        "source_last_epoch": 30,
        "source_best_map50_95": 0.253237785,
        "best_sha256": sha256(checkpoint_dir / "best.pt"),
        "last_sha256": sha256(checkpoint_dir / "last.pt"),
        "seeded_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_json(target / "seed_provenance.json", provenance)


def stream_process(command: list[str], root: Path, log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        return process.wait()


def create_final_candidate(
    root: Path, plan_name: str, experiments: list[str]
) -> dict[str, Any]:
    selected = choose_global_best(root, experiments)
    output = root / "results" / "home_final_search" / plan_name / "final_candidate"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["checkpoint"], output / "best.pt")
    experiment_dir = root / "results" / "training" / str(selected["experiment"])
    shutil.copy2(experiment_dir / "config.json", output / "config.json")
    shutil.copy2(experiment_dir / "history.csv", output / "history.csv")
    best_row = dict(selected["best_row"])
    metadata = {
        "selection_metric": "Valid mAP50:95",
        "test_split_used": False,
        "selected_experiment": selected["experiment"],
        "selected_epoch": selected["epoch"],
        "map50_95": selected["map50_95"],
        "ap75": selected["ap75"],
        "recall": selected["recall"],
        "small_recall": selected["small_recall"],
        "best_row": best_row,
        "checkpoint_sha256": sha256(output / "best.pt"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_json(output / "selection.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the safe three-stage home final performance search."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--plan", type=Path, default=Path("plans/home_final_search.json")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    plan_path = resolve_from_root(root, args.plan)
    plan = load_json(plan_path)
    stages = validate_plan(root, plan)
    plan_name = str(plan["plan_name"])
    seed = resolve_from_root(root, str(plan["seed_directory"]))
    result_dir = root / "results" / "home_final_search" / plan_name
    state_path = result_dir / "search_state.json"
    log_path = result_dir / "search_runner.log"

    print("=" * 92)
    print("HOME FINAL PERFORMANCE SEARCH")
    print("=" * 92)
    print(f"Plan            : {plan_name}")
    print(f"Reference mAP   : {float(plan['reference_map50_95']):.6f}")
    print(f"Total GPU budget: {float(plan['total_budget_hours']):.1f} hours")
    for index, stage in enumerate(stages, 1):
        config = stage["config"]
        print(
            f"Stage {index}         : {stage['name']} | mode={stage['mode']} | "
            f"lr={float(config['learning_rate']):.6f} | "
            f"max_epoch={int(config['epochs'])} | budget={stage['budget_hours']:.1f} h"
        )
    if args.dry_run:
        print("STATUS          : DRY-RUN PASS")
        return 0

    dataset = root / "data" / "processed" / "v1_grouped"
    for split in ("train", "valid"):
        if not (dataset / split / "images").is_dir() or not (
            dataset / split / "labels"
        ).is_dir():
            raise FileNotFoundError(f"Dataset split is incomplete: {dataset / split}")

    result_dir.mkdir(parents=True, exist_ok=True)
    if state_path.is_file():
        state = load_json(state_path)
        if state.get("plan_name") != plan_name:
            raise ValueError("Existing state belongs to another plan")
    else:
        state = {
            "plan_name": plan_name,
            "plan": str(plan_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "stages": {},
        }
        atomic_json(state_path, state)

    seed_first_stage(root, seed, stages[0]["experiment"])
    completed_experiments: list[str] = []
    max_retries = int(plan.get("max_retries_per_stage", 2))
    retry_delay = int(plan.get("retry_delay_seconds", 30))

    for index, stage in enumerate(stages):
        experiment = str(stage["experiment"])
        experiment_dir = root / "results" / "training" / experiment
        checkpoint_dir = experiment_dir / "checkpoints"
        last_checkpoint = checkpoint_dir / "last.pt"
        stage_state = state["stages"].setdefault(
            stage["name"],
            {
                "experiment": experiment,
                "config": str(stage["config_path"]),
                "used_seconds": 0.0,
                "status": "pending",
            },
        )

        if stage_state.get("status") == "completed":
            print(f"SKIP completed stage: {stage['name']}")
            completed_experiments.append(experiment)
            continue

        budget_seconds = float(stage["budget_hours"]) * 3600.0
        return_code = 1
        for attempt in range(max_retries + 1):
            remaining = budget_seconds - float(stage_state.get("used_seconds", 0.0))
            if remaining <= 60:
                if last_checkpoint.is_file():
                    print(f"Stage budget exhausted; preserving {last_checkpoint}")
                    return_code = 0
                    break
                raise RuntimeError(f"Stage {stage['name']} exhausted budget without checkpoint")

            command = [
                sys.executable,
                "-u",
                str(root / "scripts" / "16_train.py"),
                "--root",
                str(root),
                "--config",
                str(stage["config_path"]),
                "--max-runtime-hours",
                f"{remaining / 3600.0:.6f}",
            ]
            mode = "fresh"
            source: Path | None = None
            if last_checkpoint.is_file():
                checkpoint_epoch = validate_resume_compatibility(
                    last_checkpoint, stage["config"]
                )
                sanitize_history_for_resume(
                    experiment_dir / "history.csv", checkpoint_epoch
                )
                command.extend(["--resume", str(last_checkpoint)])
                mode = "resume"
                source = last_checkpoint
            else:
                prior_names = [str(item["experiment"]) for item in stages[:index]]
                parent = choose_global_best(root, prior_names)
                source = Path(parent["checkpoint"])
                command.extend(["--init-weights", str(source)])
                mode = f"weights_only_global_best:{parent['experiment']}@{parent['epoch']}"

            banner = (
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"stage={stage['name']} attempt={attempt + 1} mode={mode} "
                f"remaining_hours={remaining / 3600.0:.2f}"
            )
            print("-" * 92)
            print(banner, flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(banner + "\n")
                if source is not None:
                    log.write(f"source={source}\n")
            stage_state["status"] = "running"
            stage_state["last_started_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_json(state_path, state)
            started = time.time()
            try:
                return_code = stream_process(command, root, log_path)
            except KeyboardInterrupt:
                stage_state["used_seconds"] = float(stage_state["used_seconds"]) + (
                    time.time() - started
                )
                stage_state["status"] = "user_interrupted"
                atomic_json(state_path, state)
                print("User interrupted. Run the same command to resume from last.pt.")
                return 130
            stage_state["used_seconds"] = float(stage_state["used_seconds"]) + (
                time.time() - started
            )
            stage_state["last_return_code"] = return_code
            atomic_json(state_path, state)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"return_code={return_code}\n")
            if return_code == 0:
                break
            if attempt < max_retries:
                print(f"Unexpected exit; retrying in {retry_delay} seconds...", flush=True)
                time.sleep(retry_delay)

        if return_code != 0:
            stage_state["status"] = "failed"
            state["status"] = "failed"
            atomic_json(state_path, state)
            return return_code
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Training returned success without last.pt: {experiment}")
        candidate = best_record(root, experiment)
        if candidate is None:
            raise FileNotFoundError(f"Training returned success without best result: {experiment}")
        stage_state.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "last_checkpoint": str(last_checkpoint),
                "best_checkpoint": str(candidate["checkpoint"]),
                "best_epoch": candidate["epoch"],
                "best_map50_95": candidate["map50_95"],
            }
        )
        completed_experiments.append(experiment)
        global_best = choose_global_best(root, completed_experiments)
        stage_state["global_best_after_stage"] = {
            "experiment": global_best["experiment"],
            "epoch": global_best["epoch"],
            "map50_95": global_best["map50_95"],
        }
        atomic_json(state_path, state)
        print(
            f"Stage complete: {stage['name']} | stage best={candidate['map50_95']:.6f} | "
            f"global best={global_best['map50_95']:.6f}"
        )

    metadata = create_final_candidate(root, plan_name, completed_experiments)
    state["status"] = "completed"
    state["completed_at"] = datetime.now().isoformat(timespec="seconds")
    state["final_candidate"] = metadata
    atomic_json(state_path, state)
    print("=" * 92)
    print("HOME FINAL SEARCH COMPLETE")
    print(f"Selected experiment : {metadata['selected_experiment']}")
    print(f"Selected epoch      : {metadata['selected_epoch']}")
    print(f"Valid mAP50:95      : {float(metadata['map50_95']):.6f}")
    print(f"Final checkpoint    : {result_dir / 'final_candidate' / 'best.pt'}")
    print("Test split          : NOT USED")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
