from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the complete detector forward path.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timing-runs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import PersonDetector, count_trainable_parameters

    torch.manual_seed(20260811)
    model = PersonDetector(fpn_channels=64, backbone_expansion=2.0)
    inputs = torch.randn(args.batch_size, 3, 240, 320)
    spatial_shapes = {
        "p2": (60, 80),
        "p3": (30, 40),
        "p4": (15, 20),
        "p5": (8, 10),
    }
    errors: list[str] = []

    model.eval()
    with torch.inference_mode():
        predictions = model(inputs)
    for level, spatial_shape in spatial_shapes.items():
        expected = {
            "class_logits": (args.batch_size, 1, *spatial_shape),
            "quality_logits": (args.batch_size, 1, *spatial_shape),
            "distances": (args.batch_size, 4, *spatial_shape),
        }
        for name, tensor in predictions[level].items():
            if tuple(tensor.shape) != expected[name]:
                errors.append(
                    f"{level}/{name}: expected {expected[name]}, found {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                errors.append(f"{level}/{name}: non-finite values")
        if torch.any(predictions[level]["distances"] <= 0):
            errors.append(f"{level}/distances: regression distance must be positive")

    model.train()
    train_predictions = model(torch.randn(1, 3, 240, 320))
    loss = sum(
        output["class_logits"].sigmoid().mean()
        + output["quality_logits"].sigmoid().mean()
        + output["distances"].mean() * 0.001
        for output in train_predictions.values()
    )
    loss.backward()
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        errors.append(f"Parameters without gradients: {missing_gradients[:10]}")
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        errors.append(f"Non-finite gradients: {nonfinite_gradients[:10]}")

    model.eval()
    timing_input = torch.randn(1, 3, 240, 320)
    with torch.inference_mode():
        for _ in range(5):
            model(timing_input)
        timings_ms: list[float] = []
        for _ in range(args.timing_runs):
            start = time.perf_counter()
            model(timing_input)
            timings_ms.append((time.perf_counter() - start) * 1000)

    parameters = count_trainable_parameters(model)
    state_size_mb = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    ) / (1024**2)
    total_locations = sum(height * width for height, width in spatial_shapes.values())

    print("=" * 72)
    print("ANCHOR-FREE DETECTION HEAD TEST")
    print("=" * 72)
    print(f"Input            : {tuple(inputs.shape)}")
    for level, output in predictions.items():
        print(f"\n{level.upper()}")
        for name, tensor in output.items():
            print(f"  {name:<14}: {tuple(tensor.shape)}")
    print(f"\nLocations/image  : {total_locations:,}")
    print(f"Total parameters : {parameters:,}")
    print(f"State size FP32  : {state_size_mb:.2f} MiB")
    print(f"CPU median       : {statistics.median(timings_ms):.2f} ms/image")
    print(f"Backward loss    : {loss.item():.6f}")
    print(f"Total errors     : {len(errors)}")
    print(f"STATUS           : {'PASS' if not errors else 'FAIL'}")
    print("=" * 72)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
