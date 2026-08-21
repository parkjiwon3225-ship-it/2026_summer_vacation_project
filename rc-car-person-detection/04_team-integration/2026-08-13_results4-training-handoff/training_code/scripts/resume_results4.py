from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HANDOFF_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CODE_ROOT = HANDOFF_ROOT / "training_code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume the results.4 FPN48 experiment with its saved optimizer state."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Project root containing data/processed/v1_grouped.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("last", "best"),
        default="last",
        help="Use last for exact epoch-49 continuation; best restarts from epoch 38.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    required = [
        data_root / "data" / "processed" / "v1_grouped" / split / kind
        for split in ("train", "valid")
        for kind in ("images", "labels")
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Required grouped dataset folders are missing:\n- " + "\n- ".join(missing)
        )

    config_path = TRAINING_CODE_ROOT / "configs" / "school2_lightweight_100e.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment_name"] = (
        "results4_fpn48_continued_from_epoch48"
        if args.checkpoint == "last"
        else "results4_fpn48_branch_from_best_epoch37"
    )
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    checkpoint_path = HANDOFF_ROOT / "model" / f"results4_fpn48_{args.checkpoint}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(TRAINING_CODE_ROOT / "src"))
    from rc_detector.training import run_training

    run_training(data_root, config, checkpoint_path)


if __name__ == "__main__":
    main()
