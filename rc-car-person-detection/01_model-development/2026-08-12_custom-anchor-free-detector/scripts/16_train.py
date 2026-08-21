from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom RC person detector.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-runtime-hours", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector.training import run_training

    config_path = (args.config or root / "configs" / "train_v1.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.max_runtime_hours is not None:
        if args.max_runtime_hours <= 0:
            raise ValueError("--max-runtime-hours must be positive")
        config["max_runtime_hours"] = args.max_runtime_hours
    if args.resume is not None and args.init_weights is not None:
        raise ValueError("--resume and --init-weights cannot be used together")
    resume_path = args.resume.resolve() if args.resume else None
    initial_weights_path = args.init_weights.resolve() if args.init_weights else None
    run_training(
        root,
        config,
        resume_path=resume_path,
        initial_weights_path=initial_weights_path,
    )


if __name__ == "__main__":
    main()
