from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test object detection metrics.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import DetectionEvaluator

    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [50.0, 40.0, 90.0, 100.0]])},
        {"boxes": torch.tensor([[100.0, 50.0, 130.0, 150.0]])},
    ]
    perfect_predictions = [
        {
            "boxes": targets[0]["boxes"].clone(),
            "scores": torch.tensor([0.95, 0.90]),
        },
        {
            "boxes": targets[1]["boxes"].clone(),
            "scores": torch.tensor([0.85]),
        },
    ]
    perfect = DetectionEvaluator(operating_score_threshold=0.25)
    perfect.update(perfect_predictions, targets)
    perfect_metrics = perfect.compute()
    errors: list[str] = []
    for key in ("ap50", "ap75", "map50_95", "precision", "recall", "f1"):
        if abs(float(perfect_metrics[key]) - 1.0) > 1e-6:
            errors.append(f"Perfect case {key} expected 1, found {perfect_metrics[key]}")

    imperfect_predictions = [
        {
            "boxes": torch.tensor(
                [
                    [10.0, 10.0, 30.0, 30.0],
                    [150.0, 150.0, 180.0, 180.0],
                ]
            ),
            "scores": torch.tensor([0.90, 0.80]),
        },
        {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,))},
    ]
    imperfect = DetectionEvaluator(operating_score_threshold=0.25)
    imperfect.update(imperfect_predictions, targets)
    imperfect_metrics = imperfect.compute()
    expected = {"tp": 1, "fp": 1, "fn": 2}
    for key, value in expected.items():
        if imperfect_metrics[key] != value:
            errors.append(f"Imperfect case {key} expected {value}, found {imperfect_metrics[key]}")
    if abs(float(imperfect_metrics["precision"]) - 0.5) > 1e-6:
        errors.append("Imperfect precision is incorrect")
    if abs(float(imperfect_metrics["recall"]) - 1 / 3) > 1e-6:
        errors.append("Imperfect recall is incorrect")
    if abs(float(imperfect_metrics["f1"]) - 0.4) > 1e-6:
        errors.append("Imperfect F1 is incorrect")

    print("=" * 72)
    print("DETECTION METRICS TEST")
    print("=" * 72)
    print(f"Perfect AP50       : {perfect_metrics['ap50']:.6f}")
    print(f"Perfect mAP50:95   : {perfect_metrics['map50_95']:.6f}")
    print(f"Perfect P/R/F1     : {perfect_metrics['precision']:.3f} / {perfect_metrics['recall']:.3f} / {perfect_metrics['f1']:.3f}")
    print(f"Imperfect TP/FP/FN : {imperfect_metrics['tp']} / {imperfect_metrics['fp']} / {imperfect_metrics['fn']}")
    print(f"Imperfect P/R/F1   : {imperfect_metrics['precision']:.3f} / {imperfect_metrics['recall']:.3f} / {imperfect_metrics['f1']:.3f}")
    print(f"Detection accuracy : {imperfect_metrics['detection_accuracy']:.3f}")
    print(f"Total errors       : {len(errors)}")
    print(f"STATUS             : {'PASS' if not errors else 'FAIL'}")
    print("=" * 72)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
