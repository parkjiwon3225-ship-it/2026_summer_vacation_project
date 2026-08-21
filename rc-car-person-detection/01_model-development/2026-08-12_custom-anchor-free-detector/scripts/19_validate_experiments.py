from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate six GPU experiment configs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import AnchorFreeTargetAssigner, DetectionLoss, PersonDetector

    config_paths = sorted((root / "configs" / "experiments").glob("gpu*.json"))
    errors: list[str] = []
    if len(config_paths) != 6:
        errors.append(f"Expected 6 configs, found {len(config_paths)}")
    names: set[str] = set()
    print("=" * 100)
    print("SIX-GPU EXPERIMENT CONFIG VALIDATION")
    print("=" * 100)
    print(f"{'CONFIG':<26} {'LR':>8} {'FPN':>6} {'EXP':>6} {'BOX-W':>7} {'RADIUS':>8} {'PARAMS':>12}")
    for path in config_paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        name = str(config["experiment_name"])
        if name in names:
            errors.append(f"Duplicate experiment name: {name}")
        names.add(name)
        model = PersonDetector(
            fpn_channels=int(config["fpn_channels"]),
            backbone_expansion=float(config["backbone_expansion"]),
        )
        DetectionLoss(
            assigner=AnchorFreeTargetAssigner(
                center_sampling_radius=float(config["center_sampling_radius"])
            ),
            box_weight=float(config["box_loss_weight"]),
            quality_weight=float(config["quality_loss_weight"]),
        )
        with torch.inference_mode():
            outputs = model.eval()(torch.zeros(1, 3, 240, 320))
        if set(outputs) != {"p2", "p3", "p4", "p5"}:
            errors.append(f"{path.name}: invalid model output levels")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{path.name:<26} {config['learning_rate']:>8.4g} "
            f"{config['fpn_channels']:>6} {config['backbone_expansion']:>6.1f} "
            f"{config['box_loss_weight']:>7.1f} {config['center_sampling_radius']:>8.1f} "
            f"{parameters:>12,}"
        )
    print("-" * 100)
    print(f"Total errors: {len(errors)}")
    print(f"STATUS      : {'PASS' if not errors else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
