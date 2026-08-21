from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test custom backbone and FPN.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timing-runs", type=int, default=20)
    return parser.parse_args()


class BackboneWithFPN(nn.Module):
    def __init__(self, backbone: nn.Module, fpn: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.fpn(self.backbone(inputs))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import (
        LightweightBackbone,
        LightweightFPN,
        count_trainable_parameters,
    )

    torch.manual_seed(20260811)
    backbone = LightweightBackbone(expansion=2.0)
    fpn = LightweightFPN(dict(backbone.output_channels), out_channels=64)
    model = BackboneWithFPN(backbone, fpn)
    inputs = torch.randn(args.batch_size, 3, 240, 320)
    expected_shapes = {
        "p2": (args.batch_size, 64, 60, 80),
        "p3": (args.batch_size, 64, 30, 40),
        "p4": (args.batch_size, 64, 15, 20),
        "p5": (args.batch_size, 64, 8, 10),
    }
    errors: list[str] = []
    model.eval()
    with torch.inference_mode():
        outputs = model(inputs)
    for name, expected in expected_shapes.items():
        actual = tuple(outputs[name].shape)
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
        if not torch.isfinite(outputs[name]).all():
            errors.append(f"{name}: non-finite output")

    model.train()
    gradient_outputs = model(torch.randn(1, 3, 240, 320))
    loss = sum(feature.abs().mean() for feature in gradient_outputs.values())
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

    backbone_parameters = count_trainable_parameters(backbone)
    fpn_parameters = count_trainable_parameters(fpn)
    total_parameters = count_trainable_parameters(model)
    state_size_mb = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    ) / (1024**2)

    print("=" * 72)
    print("LIGHTWEIGHT BACKBONE + FPN TEST")
    print("=" * 72)
    print(f"Input          : {tuple(inputs.shape)}")
    for name, feature in outputs.items():
        print(f"{name.upper():<14} : {tuple(feature.shape)}")
    print(f"Backbone params: {backbone_parameters:,}")
    print(f"FPN params     : {fpn_parameters:,}")
    print(f"Total params   : {total_parameters:,}")
    print(f"State size FP32: {state_size_mb:.2f} MiB")
    print(f"CPU median     : {statistics.median(timings_ms):.2f} ms/image")
    print(f"Backward loss  : {loss.item():.6f}")
    print(f"Total errors   : {len(errors)}")
    print(f"STATUS         : {'PASS' if not errors else 'FAIL'}")
    print("=" * 72)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
