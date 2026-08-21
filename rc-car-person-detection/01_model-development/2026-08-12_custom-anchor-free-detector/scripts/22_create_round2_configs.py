from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = {
    "seed",
    "image_width",
    "image_height",
    "epochs",
    "batch_size",
    "num_workers",
    "learning_rate",
    "fpn_channels",
    "backbone_expansion",
    "box_loss_weight",
    "quality_loss_weight",
    "center_sampling_radius",
    "experiment_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create validated Round 2 configs from a reviewed plan.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_config(config: dict[str, Any], source: Path) -> None:
    missing = sorted(REQUIRED_KEYS - set(config))
    if missing:
        raise ValueError(f"{source}: missing keys {missing}")
    if int(config["epochs"]) <= 0 or int(config["batch_size"]) <= 0:
        raise ValueError(f"{source}: epochs and batch_size must be positive")
    if int(config["image_width"]) % 32 or int(config["image_height"]) % 16:
        raise ValueError(f"{source}: image dimensions are incompatible with the current feature pyramid")
    if float(config["learning_rate"]) <= 0:
        raise ValueError(f"{source}: learning_rate must be positive")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    experiments = plan.get("experiments", [])
    if not experiments:
        raise ValueError("Plan contains no experiments")
    if len(experiments) > 6:
        raise ValueError("Round 2 is limited to at most six experiments")
    output_dir = (args.output_dir or root / "configs" / "round2").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    names = set()
    commands = []
    written = []
    for index, experiment in enumerate(experiments, start=1):
        name = str(experiment["name"])
        if name in names:
            raise ValueError(f"Duplicate experiment name: {name}")
        names.add(name)
        base_path = (root / str(experiment["base_config"])).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        config = json.loads(base_path.read_text(encoding="utf-8"))
        config.update(plan.get("common_overrides", {}))
        config.update(experiment.get("overrides", {}))
        config["experiment_name"] = name
        validate_config(config, base_path)
        destination = output_dir / f"round2_gpu{index}_{name}.json"
        destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(destination)
        try:
            command_config = destination.relative_to(root)
        except ValueError:
            command_config = destination
        commands.append(
            f"GPU {index}: python scripts\\16_train.py --config "
            f"{str(command_config).replace('/', chr(92))}"
        )

    command_path = output_dir / "ROUND2_COMMANDS.txt"
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print("=" * 88)
    print("ROUND 2 CONFIG GENERATION")
    print("=" * 88)
    for path in written:
        config = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{path.name:<48} epochs={config['epochs']:<3} lr={config['learning_rate']:<8} "
            f"fpn={config['fpn_channels']:<3} exp={config['backbone_expansion']:<4} "
            f"box={config['box_loss_weight']:<3} radius={config['center_sampling_radius']}"
        )
    print("-" * 88)
    print(f"Commands: {command_path}")
    print("STATUS  : PASS")


if __name__ == "__main__":
    main()
