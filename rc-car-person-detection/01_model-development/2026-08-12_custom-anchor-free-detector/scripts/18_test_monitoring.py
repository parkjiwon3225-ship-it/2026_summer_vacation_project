from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test training monitoring on tiny subsets.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import DetectionLoss, DetectionPostProcessor, PersonDetectionDataset, PersonDetector
    from rc_detector.training import (
        create_grad_scaler,
        create_loader,
        evaluate_detections,
        health_warnings,
        train_one_epoch,
        validate_one_epoch,
    )

    torch.manual_seed(20260811)
    device = torch.device("cpu")
    dataset_dir = root / "data" / "processed" / "v1_grouped"
    train_full = PersonDetectionDataset(dataset_dir, "train", augment=False)
    valid_full = PersonDetectionDataset(dataset_dir, "valid", augment=False)
    train_dataset = Subset(train_full, list(range(4)))
    valid_dataset = Subset(valid_full, list(range(4)))
    train_loader = create_loader(train_dataset, 2, 0, False, 20260811, device)
    valid_loader = create_loader(valid_dataset, 2, 0, False, 20260811, device)
    model = PersonDetector().to(device)
    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = create_grad_scaler(False)
    train_metrics = train_one_epoch(
        model, criterion, train_loader, optimizer, scaler, device, False, 1, 10.0
    )
    valid_metrics = validate_one_epoch(model, criterion, valid_loader, device, False)
    detection_metrics = evaluate_detections(
        model,
        valid_loader,
        device,
        False,
        DetectionPostProcessor(score_threshold=0.0, pre_nms_topk=50, max_detections=20),
        0.25,
    )
    warnings = health_warnings(1, train_metrics, valid_metrics, detection_metrics)
    errors: list[str] = []
    required_train = {"total", "classification", "quality", "box", "gradient_norm"}
    required_detection = {"ap50", "ap75", "map50_95", "precision", "recall", "f1", "tp", "fp", "fn"}
    if not required_train.issubset(train_metrics):
        errors.append("Training monitoring fields are missing")
    if not required_detection.issubset(detection_metrics):
        errors.append("Detection monitoring fields are missing")
    if not all(torch.isfinite(torch.tensor(float(value))) for value in train_metrics.values()):
        errors.append("Non-finite training metric")

    print("=" * 72)
    print("TRAINING MONITORING SMOKE TEST")
    print("=" * 72)
    print(f"Train total/grad : {train_metrics['total']:.6f} / {train_metrics['gradient_norm']:.6f}")
    print(f"Valid total      : {valid_metrics['total']:.6f}")
    print(f"AP50 / mAP       : {detection_metrics['ap50']:.6f} / {detection_metrics['map50_95']:.6f}")
    print(f"P / R / F1       : {detection_metrics['precision']:.6f} / {detection_metrics['recall']:.6f} / {detection_metrics['f1']:.6f}")
    print(f"TP / FP / FN     : {detection_metrics['tp']} / {detection_metrics['fp']} / {detection_metrics['fn']}")
    print(f"Warnings         : {len(warnings)}")
    print(f"Total errors     : {len(errors)}")
    print(f"STATUS           : {'PASS' if not errors else 'FAIL'}")
    print("=" * 72)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
