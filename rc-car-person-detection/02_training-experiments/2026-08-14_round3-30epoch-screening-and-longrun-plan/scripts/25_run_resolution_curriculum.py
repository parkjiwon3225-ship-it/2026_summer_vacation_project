from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def resolve_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


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


def checkpoint_for_experiment(
    root: Path, experiment_name: str, preference: str
) -> Path:
    checkpoint_dir = root / "results" / "training" / experiment_name / "checkpoints"
    preferred = checkpoint_dir / f"{preference}.pt"
    fallback = checkpoint_dir / ("last.pt" if preference == "best" else "best.pt")
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"No best.pt or last.pt available for previous stage: {experiment_name}"
    )


def validate_manifest(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    required = {"plan_name", "total_budget_hours", "stages"}
    missing = required - set(manifest)
    if missing:
        raise KeyError(f"Manifest is missing: {sorted(missing)}")
    total_budget = float(manifest["total_budget_hours"])
    if not 0 < total_budget <= 100:
        raise ValueError("total_budget_hours must be between 0 and 100")
    stages = manifest["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")
    budget_sum = sum(float(stage["budget_hours"]) for stage in stages)
    if budget_sum > total_budget:
        raise ValueError(
            f"Stage budgets ({budget_sum:.2f} h) exceed total budget ({total_budget:.2f} h)"
        )
    names: set[str] = set()
    experiment_names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise TypeError(f"Stage {index + 1} is not an object")
        name = str(stage["name"])
        if name in names:
            raise ValueError(f"Duplicate stage name: {name}")
        names.add(name)
        config_path = resolve_from_root(root, str(stage["config"]))
        config = load_json(config_path)
        experiment_name = str(config["experiment_name"])
        if experiment_name in experiment_names:
            raise ValueError(f"Duplicate experiment_name: {experiment_name}")
        experiment_names.add(experiment_name)
        if int(config.get("checkpoint_every", 0)) != 1:
            raise ValueError(f"{config_path.name}: checkpoint_every must be 1")
        if int(config.get("metrics_every", 0)) != 1:
            raise ValueError(f"{config_path.name}: metrics_every must be 1")
        normalized.append(
            {
                **stage,
                "name": name,
                "config_path": config_path,
                "config": config,
                "budget_hours": float(stage["budget_hours"]),
            }
        )
    for index, stage in enumerate(normalized):
        if stage.get("transfer_from_previous"):
            if index == 0:
                raise ValueError("First stage cannot transfer from a previous stage")
            previous = normalized[index - 1]["config"]
            current = stage["config"]
            for key in ("fpn_channels", "backbone_expansion"):
                if current[key] != previous[key]:
                    raise ValueError(
                        f"Weights-only transfer requires matching {key}: "
                        f"{previous[key]} != {current[key]}"
                    )
    final_config = normalized[-1]["config"]
    if (int(final_config["image_width"]), int(final_config["image_height"])) != (
        320,
        240,
    ):
        raise ValueError("The final stage must train and evaluate at 320x240")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a time-bounded high-resolution to 320x240 curriculum."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = resolve_from_root(root, args.manifest)
    manifest = load_json(manifest_path)
    stages = validate_manifest(root, manifest)
    plan_name = str(manifest["plan_name"])
    max_retries = int(manifest.get("max_retries_per_stage", 2))
    retry_delay = int(manifest.get("retry_delay_seconds", 30))
    result_dir = root / "results" / "longrun" / plan_name
    state_path = result_dir / "longrun_state.json"
    log_path = result_dir / "longrun_runner.log"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("100-HOUR RESOLUTION CURRICULUM")
    print("=" * 88)
    print(f"Plan           : {plan_name}")
    print(f"Manifest       : {manifest_path}")
    print(f"Total budget   : {float(manifest['total_budget_hours']):.1f} hours")
    for index, stage in enumerate(stages, 1):
        config = stage["config"]
        print(
            f"Stage {index:<2}       : {stage['name']} | "
            f"{config['image_width']}x{config['image_height']} | "
            f"max {config['epochs']} epoch | {stage['budget_hours']:.1f} h"
        )
    if args.dry_run:
        print("STATUS         : DRY-RUN PASS")
        return 0

    now = time.time()
    if state_path.is_file():
        state = load_json(state_path)
        if state.get("plan_name") != plan_name:
            raise ValueError("Existing state belongs to a different plan")
    else:
        state = {
            "plan_name": plan_name,
            "manifest": str(manifest_path),
            "global_started_at_epoch": now,
            "global_started_at": datetime.now().isoformat(timespec="seconds"),
            "current_stage_index": 0,
            "stages": {},
            "status": "running",
        }
        atomic_json(state_path, state)

    global_deadline = float(state["global_started_at_epoch"]) + float(
        manifest["total_budget_hours"]
    ) * 3600.0

    for stage_index, stage in enumerate(stages):
        stage_name = stage["name"]
        config = stage["config"]
        experiment_name = str(config["experiment_name"])
        experiment_dir = root / "results" / "training" / experiment_name
        checkpoint = experiment_dir / "checkpoints" / "last.pt"
        stage_state = state["stages"].setdefault(stage_name, {})
        stage_state.setdefault("experiment_name", experiment_name)
        stage_state.setdefault("config", str(stage["config_path"]))
        if stage_state.get("status") == "completed":
            print(f"SKIP completed stage: {stage_name}")
            continue

        if "started_at_epoch" not in stage_state:
            stage_state["started_at_epoch"] = time.time()
            stage_state["started_at"] = datetime.now().isoformat(timespec="seconds")
        state["current_stage_index"] = stage_index
        atomic_json(state_path, state)

        later_budget_seconds = sum(
            float(later["budget_hours"]) * 3600.0
            for later in stages[stage_index + 1 :]
        )
        stage_deadline = min(
            float(stage_state["started_at_epoch"]) + stage["budget_hours"] * 3600.0,
            global_deadline - later_budget_seconds,
        )

        training_status_path = experiment_dir / "training_status.json"
        if training_status_path.is_file():
            previous_status = load_json(training_status_path)
            if previous_status.get("stop_reason") in {
                "epochs_complete",
                "runtime_limit",
            } and checkpoint.is_file():
                stage_state.update(
                    {
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                        "training_status": previous_status,
                    }
                )
                atomic_json(state_path, state)
                print(f"RECOVER completed stage from status file: {stage_name}")
                continue

        return_code = 1
        for attempt in range(max_retries + 1):
            remaining_seconds = min(stage_deadline, global_deadline) - time.time()
            if remaining_seconds <= 60:
                if checkpoint.is_file():
                    print(f"Stage time budget exhausted; keeping checkpoint: {checkpoint}")
                    return_code = 0
                    break
                raise RuntimeError(
                    f"Stage {stage_name} exhausted its time budget without a checkpoint"
                )

            command = [
                sys.executable,
                "-u",
                str(root / "scripts" / "16_train.py"),
                "--root",
                str(root),
                "--config",
                str(stage["config_path"]),
                "--max-runtime-hours",
                f"{remaining_seconds / 3600.0:.6f}",
            ]
            mode = "fresh"
            source: Path | None = None
            if checkpoint.is_file():
                command.extend(["--resume", str(checkpoint)])
                mode = "resume"
                source = checkpoint
            else:
                initial_weights = stage.get("initial_weights")
                transfer = stage.get("transfer_from_previous")
                if initial_weights:
                    source = resolve_from_root(root, str(initial_weights))
                    if not source.is_file():
                        raise FileNotFoundError(f"Initial weights are missing: {source}")
                    command.extend(["--init-weights", str(source)])
                    mode = "weights_only_file"
                elif transfer:
                    if stage_index == 0:
                        raise ValueError("First stage cannot transfer from a previous stage")
                    previous_experiment = str(
                        stages[stage_index - 1]["config"]["experiment_name"]
                    )
                    preference = str(stage.get("checkpoint_preference", "best"))
                    source = checkpoint_for_experiment(
                        root, previous_experiment, preference
                    )
                    command.extend(["--init-weights", str(source)])
                    mode = f"weights_only_previous_{preference}"

            banner = (
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"stage={stage_name} attempt={attempt + 1} mode={mode} "
                f"remaining_hours={remaining_seconds / 3600.0:.2f}"
            )
            print("-" * 88)
            print(banner, flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(banner + "\n")
                if source is not None:
                    log.write(f"source={source}\n")
            try:
                return_code = stream_process(command, root, log_path)
            except KeyboardInterrupt:
                stage_state["status"] = "user_interrupted"
                atomic_json(state_path, state)
                print("User interrupted. Last completed epoch remains safe.")
                return 130
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"return_code={return_code}\n")
            if return_code == 0:
                break
            if attempt < max_retries:
                print(
                    f"Unexpected exit. Retrying after {retry_delay} seconds...",
                    flush=True,
                )
                time.sleep(retry_delay)

        if return_code != 0:
            stage_state.update(
                {
                    "status": "failed",
                    "return_code": return_code,
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            state["status"] = "failed"
            atomic_json(state_path, state)
            print(f"FAILED stage: {stage_name}")
            return return_code

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Stage returned success but last.pt is missing: {checkpoint}"
            )
        stage_state.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "checkpoint": str(checkpoint),
            }
        )
        if training_status_path.is_file():
            stage_state["training_status"] = load_json(training_status_path)
        atomic_json(state_path, state)

    state["status"] = "completed"
    state["completed_at"] = datetime.now().isoformat(timespec="seconds")
    state["elapsed_hours"] = (
        time.time() - float(state["global_started_at_epoch"])
    ) / 3600.0
    atomic_json(state_path, state)
    print("=" * 88)
    print("LONG-RUN PLAN COMPLETE")
    print(f"State folder    : {result_dir}")
    print(f"Elapsed hours   : {state['elapsed_hours']:.2f}")
    print("Final stage is 320x240 and ready for Valid comparison.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
