from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_BEST_SHA256 = "D44DB15BCB623EDE678C48150CE7EF6965F871FA0D3DF80D8BEC2E900F8FF27A"
EXPECTED_LAST_SHA256 = "1E4633F06FE33E91F069DDF1E8BF9EBFE2487A75561EBC764BD854345A9B12FA"
EXPECTED_MAP = 0.253237785


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def get_seed_dir(root: Path) -> Path:
    plan_path = root / "plans" / "home_final_search.json"
    plan = load_json(plan_path)
    seed_directory = plan.get("seed_directory")
    if not isinstance(seed_directory, str) or not seed_directory.strip():
        raise ValueError("plan seed_directory is missing")
    return (root / seed_directory).resolve()


def verify_manifest(root: Path, errors: list[str]) -> int:
    manifest_path = root / "PACKAGE_SHA256_MANIFEST.json"
    if not manifest_path.is_file():
        return 0
    manifest = load_json(manifest_path)
    if manifest.get("package_name") != "RC_HOME_FINAL_SEARCH_v1":
        return 0
    files = manifest.get("files", [])
    for item in files:
        path = root / str(item["path"])
        if not path.is_file():
            errors.append(f"manifest file missing: {item['path']}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            errors.append(f"manifest size mismatch: {item['path']}")
        if sha256(path) != str(item["sha256"]).upper():
            errors.append(f"manifest SHA-256 mismatch: {item['path']}")
    return len(files)


def verify_structure(root: Path, errors: list[str]) -> None:
    try:
        seed = get_seed_dir(root)
    except Exception as exc:
        errors.append(f"cannot resolve seed directory: {exc}")
        seed = root / "seeds" / "results14_seed11"
    required = (
        root / "scripts" / "16_train.py",
        root / "scripts" / "28_run_home_final_search.py",
        root / "scripts" / "29_summarize_home_final_search.py",
        root / "src" / "rc_detector" / "model.py",
        root / "src" / "rc_detector" / "training.py",
        root / "plans" / "home_final_search.json",
        root / "configs" / "home_final" / "stage1_continue_results14_to100.json",
        root / "configs" / "home_final" / "stage2_finetune_lr0250_40e.json",
        root / "configs" / "home_final" / "stage3_polish_lr0100_24e.json",
        seed / "checkpoints" / "best.pt",
        seed / "checkpoints" / "last.pt",
        seed / "history.csv",
    )
    for path in required:
        if not path.is_file():
            errors.append(f"required file missing: {path}")

    relevant_roots = (
        root / "configs" / "home_final",
        root / "plans",
        root / "scripts",
        root / "src" / "rc_detector",
        seed,
    )
    relevant_files = [
        path
        for search_root in relevant_roots
        if search_root.exists()
        for path in search_root.rglob("*")
        if path.is_file()
    ]
    for path in (item for item in relevant_files if item.suffix.lower() == ".json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid JSON {path}: {exc}")
    for path in (item for item in relevant_files if item.suffix.lower() == ".py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            errors.append(f"invalid Python {path}: {exc}")


def verify_seed(root: Path, errors: list[str]) -> None:
    seed = get_seed_dir(root)
    best = seed / "checkpoints" / "best.pt"
    last = seed / "checkpoints" / "last.pt"
    if best.is_file() and sha256(best) != EXPECTED_BEST_SHA256:
        errors.append("results.14 best.pt SHA-256 mismatch")
    if last.is_file() and sha256(last) != EXPECTED_LAST_SHA256:
        errors.append("results.14 last.pt SHA-256 mismatch")
    history_path = seed / "history.csv"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 30:
            errors.append(f"seed history must have 30 epochs, found {len(rows)}")
        if rows:
            best_row = max(rows, key=lambda row: float(row["metric_map50_95"]))
            if int(float(best_row["epoch"])) != 29:
                errors.append("seed best epoch is not 29")
            if abs(float(best_row["metric_map50_95"]) - EXPECTED_MAP) > 1e-6:
                errors.append("seed best mAP50:95 mismatch")
            if int(float(rows[-1]["epoch"])) != 30:
                errors.append("seed last history epoch is not 30")

    plan_path = root / "plans" / "home_final_search.json"
    if plan_path.is_file():
        plan = load_json(plan_path)
        stages = plan.get("stages", [])
        if [stage.get("mode") for stage in stages] != [
            "resume_seeded",
            "init_global_best",
            "init_global_best",
        ]:
            errors.append("unsafe stage modes in plan")
        if abs(sum(float(stage["budget_hours"]) for stage in stages) - float(plan["total_budget_hours"])) > 1e-6:
            errors.append("stage budget sum mismatch")


def verify_runtime(root: Path, errors: list[str]) -> None:
    try:
        import torch
    except Exception as exc:
        errors.append(f"Torch import failed: {exc}")
        return
    sys.path.insert(0, str(root / "src"))
    try:
        from rc_detector.model import PersonDetector

        model = PersonDetector(fpn_channels=48, backbone_expansion=2.0)
        seed = get_seed_dir(root)
        for filename, expected_epoch in (("best.pt", 29), ("last.pt", 30)):
            path = seed / "checkpoints" / filename
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            if int(checkpoint["epoch"]) != expected_epoch:
                errors.append(f"{filename} epoch mismatch")
            model.load_state_dict(checkpoint["model"], strict=True)
    except Exception as exc:
        errors.append(f"Checkpoint/model strict-load failed: {exc}")
    if not torch.cuda.is_available():
        errors.append("CUDA is not available")

    dataset = root / "data" / "processed" / "v1_grouped"
    expected_counts = {"train": 12322, "valid": 1531}
    for split, expected in expected_counts.items():
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"dataset split missing: {split}")
            continue
        images = sum(1 for path in image_dir.iterdir() if path.is_file())
        labels = sum(1 for path in label_dir.iterdir() if path.is_file())
        if images != expected or labels != expected:
            errors.append(
                f"{split} count mismatch: images={images}, labels={labels}, expected={expected}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the home final-search package.")
    parser.add_argument("--root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    manifest_files = verify_manifest(root, errors)
    verify_structure(root, errors)
    verify_seed(root, errors)
    if args.runtime:
        verify_runtime(root, errors)

    print("=" * 76)
    print("HOME FINAL SEARCH VERIFICATION")
    print("=" * 76)
    print(f"Root           : {root}")
    print(f"Manifest files : {manifest_files if manifest_files else 'not present / installed overlay'}")
    print(f"Runtime checks : {'enabled' if args.runtime else 'structural only'}")
    print(f"Errors         : {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"STATUS         : {'PASS' if not errors else 'CHECK REQUIRED'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
