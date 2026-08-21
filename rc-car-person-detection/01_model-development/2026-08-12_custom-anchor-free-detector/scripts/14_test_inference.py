from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test detector decode and pure PyTorch NMS.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import DetectionPostProcessor, PersonDetector, pure_torch_nms

    errors: list[str] = []
    synthetic_boxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 49.0, 49.0],
            [100.0, 100.0, 130.0, 140.0],
        ]
    )
    synthetic_scores = torch.tensor([0.90, 0.80, 0.70])
    kept = pure_torch_nms(synthetic_boxes, synthetic_scores, iou_threshold=0.5)
    if kept.tolist() != [0, 2]:
        errors.append(f"Synthetic NMS expected [0, 2], found {kept.tolist()}")

    torch.manual_seed(20260811)
    model = PersonDetector().eval()
    inputs = torch.randn(2, 3, 240, 320)
    with torch.inference_mode():
        predictions = model(inputs)
        strict_results = DetectionPostProcessor(score_threshold=0.05)(predictions)
        permissive_results = DetectionPostProcessor(
            score_threshold=0.0,
            pre_nms_topk=200,
            max_detections=50,
        )(predictions)

    for result_name, results in (
        ("strict", strict_results),
        ("permissive", permissive_results),
    ):
        if len(results) != len(inputs):
            errors.append(f"{result_name}: result batch length mismatch")
        for image_index, result in enumerate(results):
            boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                errors.append(f"{result_name}/{image_index}: invalid box shape")
            if not (len(boxes) == len(scores) == len(labels) == len(result["levels"])):
                errors.append(f"{result_name}/{image_index}: output length mismatch")
            if len(boxes):
                if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                    errors.append(f"{result_name}/{image_index}: non-finite output")
                if torch.any(boxes[:, 0] < 0) or torch.any(boxes[:, 2] > 320):
                    errors.append(f"{result_name}/{image_index}: horizontal box overflow")
                if torch.any(boxes[:, 1] < 0) or torch.any(boxes[:, 3] > 240):
                    errors.append(f"{result_name}/{image_index}: vertical box overflow")
                if torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(boxes[:, 3] <= boxes[:, 1]):
                    errors.append(f"{result_name}/{image_index}: degenerate box")
                if torch.any(scores[1:] > scores[:-1]):
                    errors.append(f"{result_name}/{image_index}: scores are not descending")

    print("=" * 72)
    print("DECODE + PURE PYTORCH NMS TEST")
    print("=" * 72)
    print(f"Synthetic NMS kept : {kept.tolist()}")
    print(f"Strict detections  : {[len(result['boxes']) for result in strict_results]}")
    print(f"Debug detections   : {[len(result['boxes']) for result in permissive_results]}")
    print(f"Total errors       : {len(errors)}")
    print(f"STATUS             : {'PASS' if not errors else 'FAIL'}")
    print("=" * 72)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
