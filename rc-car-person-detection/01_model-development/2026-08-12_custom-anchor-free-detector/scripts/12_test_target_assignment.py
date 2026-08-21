from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LEVEL_COLORS = {
    "p2": "#ffe600",
    "p3": "#00ff66",
    "p4": "#00b7ff",
    "p5": "#ff3bd4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test FCOS-style target assignment.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples", type=int, default=128)
    return parser.parse_args()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))


def validate_assignment(
    assignments: dict[str, dict[str, torch.Tensor]], boxes: torch.Tensor
) -> list[str]:
    errors: list[str] = []
    for level, target in assignments.items():
        positive = target["positive_mask"]
        quality = target["quality_targets"][0][positive]
        distances = target["distance_targets"].permute(1, 2, 0)[positive]
        matched = target["matched_gt_indices"][positive]
        points = target["points"][positive]
        if len(distances) and torch.any(distances <= 0):
            errors.append(f"{level}: positive distance is not strictly positive")
        if len(quality) and (torch.any(quality < 0) or torch.any(quality > 1)):
            errors.append(f"{level}: quality outside [0, 1]")
        if len(matched):
            matched_boxes = boxes[matched]
            reconstructed = torch.stack(
                (
                    points[:, 0] - distances[:, 0],
                    points[:, 1] - distances[:, 1],
                    points[:, 0] + distances[:, 2],
                    points[:, 1] + distances[:, 3],
                ),
                dim=1,
            )
            if not torch.allclose(reconstructed, matched_boxes, atol=1e-4):
                errors.append(f"{level}: distance target does not reconstruct GT box")
    return errors


def save_preview(
    image: torch.Tensor,
    boxes: torch.Tensor,
    assignments: dict[str, dict[str, torch.Tensor]],
    output_path: Path,
) -> None:
    preview = tensor_to_pil(image)
    draw = ImageDraw.Draw(preview)
    for box in boxes.tolist():
        draw.rectangle(box, outline="white", width=2)
    for level, target in assignments.items():
        color = LEVEL_COLORS[level]
        for x, y in target["points"][target["positive_mask"]].tolist():
            radius = 2 if level == "p3" else 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.rectangle((3, 3, 175, 48), fill="black")
    draw.text((8, 7), "white: GT box", fill="white")
    draw.text((8, 19), "yellow: P2  green: P3", fill="white")
    draw.text((8, 31), "blue: P4  pink: P5", fill="white")
    preview.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import AnchorFreeTargetAssigner, PersonDetectionDataset

    dataset_dir = root / "data" / "processed" / "v1_grouped"
    reports_dir = dataset_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset = PersonDetectionDataset(dataset_dir, "train", augment=False)
    assigner = AnchorFreeTargetAssigner()
    feature_shapes = {
        "p2": (60, 80),
        "p3": (30, 40),
        "p4": (15, 20),
        "p5": (8, 10),
    }
    sample_count = min(args.samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=int)
    total_gt = 0
    positives = {level: 0 for level in feature_shapes}
    assigned_gt: set[tuple[int, int]] = set()
    gt_sizes: dict[tuple[int, int], tuple[float, float]] = {}
    errors: list[str] = []
    preview_data = None
    for sample_number, index in enumerate(indices):
        image, target = dataset[int(index)]
        boxes, labels = target["boxes"], target["labels"]
        assignments = assigner(feature_shapes, boxes, labels)
        total_gt += len(boxes)
        for gt_index, box in enumerate(boxes.tolist()):
            gt_sizes[(sample_number, gt_index)] = (box[2] - box[0], box[3] - box[1])
        errors.extend(validate_assignment(assignments, boxes))
        sample_positive_count = 0
        for level, level_target in assignments.items():
            mask = level_target["positive_mask"]
            count = int(mask.sum())
            positives[level] += count
            sample_positive_count += count
            for gt_index in level_target["matched_gt_indices"][mask].tolist():
                assigned_gt.add((sample_number, gt_index))
        if preview_data is None and 2 <= len(boxes) <= 12 and sample_positive_count > 0:
            preview_data = (image, boxes, assignments)

    unassigned_gt = total_gt - len(assigned_gt)
    unassigned_keys = sorted(set(gt_sizes) - assigned_gt)
    unassigned_sizes = [gt_sizes[key] for key in unassigned_keys]
    unassigned_fraction = unassigned_gt / max(total_gt, 1)
    warnings: list[str] = []
    if unassigned_gt:
        warnings.append(
            f"{unassigned_gt} GT boxes ({unassigned_fraction:.2%}) received no positive location"
        )
    if unassigned_fraction > 0.02:
        errors.append(
            f"Unassigned GT rate {unassigned_fraction:.2%} exceeds the 2% limit"
        )
    empty_boxes = torch.empty((0, 4), dtype=torch.float32)
    empty_labels = torch.empty((0,), dtype=torch.int64)
    empty_assignment = assigner(feature_shapes, empty_boxes, empty_labels)
    if any(target["positive_mask"].any() for target in empty_assignment.values()):
        errors.append("Empty GT produced positive locations")

    preview_path = reports_dir / "target_assignment_preview.jpg"
    if preview_data is not None:
        save_preview(*preview_data, preview_path)
    error_path = reports_dir / "target_assignment_errors.txt"
    error_path.write_text("\n".join(errors) + ("\n" if errors else ""), encoding="utf-8")

    print("=" * 72)
    print("ANCHOR-FREE TARGET ASSIGNMENT TEST")
    print("=" * 72)
    print(f"Checked images : {sample_count}")
    print(f"GT boxes       : {total_gt}")
    for level, count in positives.items():
        print(f"{level.upper()} positives  : {count}")
    print(f"Assigned GT    : {len(assigned_gt)}")
    print(f"Unassigned GT  : {unassigned_gt}")
    print(f"Unassigned rate: {unassigned_fraction:.2%}")
    if unassigned_sizes:
        formatted_sizes = ", ".join(f"{width:.2f}x{height:.2f}" for width, height in unassigned_sizes)
        print(f"Unassigned size: {formatted_sizes}")
    print(f"Total errors   : {len(errors)}")
    print(f"Warnings       : {len(warnings)}")
    print(f"Preview        : {preview_path}")
    print(f"STATUS         : {'PASS' if not errors else 'FAIL'}")
    print("=" * 72)
    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
