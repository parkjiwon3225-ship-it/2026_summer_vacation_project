from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one real CUDA training batch at a requested resolution."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))

    import torch

    from rc_detector.assignment import AnchorFreeTargetAssigner
    from rc_detector.dataset import PersonDetectionDataset
    from rc_detector.inference import DetectionPostProcessor
    from rc_detector.losses import DetectionLoss
    from rc_detector.model import PersonDetector
    from rc_detector.training import (
        autocast_context,
        create_grad_scaler,
        create_loader,
        move_images,
        seed_everything,
    )

    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (root / args.config).resolve()
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    width, height = int(config["image_width"]), int(config["image_height"])
    batch_size = int(args.batch_size or config["batch_size"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the school-laptop smoke test")
    device = torch.device("cuda")
    seed_everything(int(config["seed"]))

    dataset = PersonDetectionDataset(
        root / "data" / "processed" / "v1_grouped",
        "train",
        image_size=(width, height),
        augment=True,
        horizontal_flip_probability=float(config["horizontal_flip_probability"]),
    )
    loader = create_loader(
        dataset,
        batch_size,
        0,
        True,
        int(config["seed"]),
        device,
    )
    model = PersonDetector(
        fpn_channels=int(config["fpn_channels"]),
        backbone_expansion=float(config["backbone_expansion"]),
    ).to(device)
    criterion = DetectionLoss(
        assigner=AnchorFreeTargetAssigner(
            center_sampling_radius=float(config["center_sampling_radius"])
        ),
        box_weight=float(config["box_loss_weight"]),
        quality_weight=float(config["quality_loss_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    amp_enabled = bool(config["amp"])
    scaler = create_grad_scaler(
        amp_enabled,
        initial_scale=float(config["amp_initial_scale"]),
        growth_interval=int(config["amp_growth_interval"]),
    )

    torch.cuda.reset_peak_memory_stats(device)
    images, targets = next(iter(loader))
    images = move_images(images, device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, amp_enabled):
        predictions = model(images)
        losses = criterion(predictions, targets)
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["gradient_clip_norm"])
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

    model.eval()
    with torch.inference_mode(), autocast_context(device, amp_enabled):
        predictions = model(images[:1])
    postprocessor = DetectionPostProcessor(
        score_threshold=float(config["evaluation_score_floor"]),
        nms_iou_threshold=float(config["nms_iou_threshold"]),
        max_detections=int(config["max_detections"]),
    )
    detections = postprocessor(predictions, image_size=(width, height))
    feature_shapes = {
        level: tuple(output["class_logits"].shape[-2:])
        for level, output in predictions.items()
    }

    print("=" * 76)
    print("RESOLUTION CUDA SMOKE TEST")
    print("=" * 76)
    print(f"Config             : {config_path}")
    print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"Input              : {width}x{height}")
    print(f"Batch size         : {batch_size}")
    print(f"Feature shapes     : {feature_shapes}")
    print(f"Loss               : {float(losses['total']):.6f}")
    print(f"Positive targets   : {int(losses['positive_count'])}")
    print(f"Gradient norm      : {float(gradient_norm):.6f}")
    print(f"Step time          : {elapsed:.3f} seconds")
    print(f"Peak allocated VRAM: {peak_vram:.3f} GiB")
    print(f"Decoded detections : {len(detections[0]['boxes'])}")
    print("STATUS             : PASS")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
