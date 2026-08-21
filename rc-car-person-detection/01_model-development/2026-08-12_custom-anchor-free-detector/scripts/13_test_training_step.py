from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test one complete detector training step.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import (
        DetectionLoss,
        PersonDetectionDataset,
        PersonDetector,
        detection_collate,
    )

    torch.manual_seed(20260811)
    dataset = PersonDetectionDataset(
        root / "data" / "processed" / "v1_grouped",
        "train",
        augment=True,
        horizontal_flip_probability=0.5,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )
    images, targets = next(iter(loader))
    model = PersonDetector()
    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    errors: list[str] = []

    model.train()
    tracked_parameter = next(model.parameters())
    parameter_before = tracked_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(images)
    losses = criterion(predictions, targets)
    if not all(torch.isfinite(value).all() for value in losses.values()):
        errors.append("Non-finite loss found")
    if losses["positive_count"].item() <= 0:
        errors.append("No positive targets in training batch")
    losses["total"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        errors.append("No gradients were produced")
    elif not all(torch.isfinite(gradient).all() for gradient in gradients):
        errors.append("Non-finite gradient found")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    if not math.isfinite(float(gradient_norm)):
        errors.append("Non-finite gradient norm")
    optimizer.step()
    parameter_delta = (tracked_parameter.detach() - parameter_before).abs().max().item()
    if parameter_delta == 0:
        errors.append("Optimizer did not update the tracked parameter")

    print("=" * 72)
    print("END-TO-END TRAINING STEP TEST")
    print("=" * 72)
    print(f"Images          : {tuple(images.shape)}")
    print(f"Boxes/image     : {[len(target['boxes']) for target in targets]}")
    print(f"Positive targets: {int(losses['positive_count'].item())}")
    print(f"Classification  : {losses['classification'].item():.6f}")
    print(f"Quality         : {losses['quality'].item():.6f}")
    print(f"GIoU box        : {losses['box'].item():.6f}")
    print(f"Total loss      : {losses['total'].item():.6f}")
    print(f"Gradient norm   : {float(gradient_norm):.6f}")
    print(f"Parameter delta : {parameter_delta:.8f}")
    print(f"Total errors    : {len(errors)}")
    print(f"STATUS          : {'PASS' if not errors else 'FAIL'}")
    print("=" * 72)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
