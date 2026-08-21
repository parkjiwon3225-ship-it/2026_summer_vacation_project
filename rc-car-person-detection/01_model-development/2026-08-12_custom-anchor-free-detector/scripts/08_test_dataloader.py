from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the custom detection DataLoader.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples-per-split", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def check_target(image: torch.Tensor, target: dict, split: str) -> list[str]:
    errors: list[str] = []
    if image.shape != (3, 240, 320):
        errors.append(f"{split}: unexpected image shape {tuple(image.shape)}")
    if image.dtype != torch.float32 or image.min() < 0 or image.max() > 1:
        errors.append(f"{split}: image values are not float32 in [0, 1]")
    boxes = target["boxes"]
    labels = target["labels"]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        errors.append(f"{split}: invalid boxes shape {tuple(boxes.shape)}")
        return errors
    if len(boxes) != len(labels):
        errors.append(f"{split}: boxes/labels count mismatch")
    if len(boxes):
        if not torch.all(labels == 0):
            errors.append(f"{split}: non-person class found")
        if torch.any(boxes[:, 0] < 0) or torch.any(boxes[:, 1] < 0):
            errors.append(f"{split}: negative box coordinate")
        if torch.any(boxes[:, 2] > 320) or torch.any(boxes[:, 3] > 240):
            errors.append(f"{split}: box exceeds letterboxed image")
        if torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(boxes[:, 3] <= boxes[:, 1]):
            errors.append(f"{split}: degenerate box")
    return errors


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))


def save_preview(samples: list[tuple[torch.Tensor, dict]], output_path: Path) -> None:
    columns, tile_width, tile_height = 4, 320, 260
    rows = (len(samples) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202124")
    for index, (image_tensor, target) in enumerate(samples):
        image = tensor_to_pil(image_tensor)
        draw = ImageDraw.Draw(image)
        for box in target["boxes"].tolist():
            draw.rectangle(box, outline="#00ff66", width=2)
        column, row = index % columns, index // columns
        canvas.paste(image, (column * tile_width, row * tile_height))
        name = Path(target["path"]).name
        if len(name) > 42:
            name = name[:39] + "..."
        ImageDraw.Draw(canvas).text(
            (column * tile_width + 4, row * tile_height + 242),
            f"{name} | n={len(target['boxes'])} | flip={target['flipped']}",
            fill="white",
        )
    canvas.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import PersonDetectionDataset, detection_collate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = root / "data" / "processed" / "v1_grouped"
    reports_dir = dataset_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    all_errors: list[str] = []
    preview_samples: list[tuple[torch.Tensor, dict]] = []

    print("=" * 72)
    print("CUSTOM DATASET / DATALOADER TEST")
    print("=" * 72)
    for split in ("train", "valid", "test"):
        dataset = PersonDetectionDataset(
            dataset_dir,
            split,
            image_size=(320, 240),
            augment=(split == "train"),
            horizontal_flip_probability=0.5,
        )
        count = min(args.samples_per_split, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
        total_boxes = 0
        split_errors: list[str] = []
        for index in indices:
            image, target = dataset[int(index)]
            total_boxes += len(target["boxes"])
            split_errors.extend(check_target(image, target, split))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=detection_collate,
        )
        batch_images, batch_targets = next(iter(loader))
        if batch_images.shape != (args.batch_size, 3, 240, 320):
            split_errors.append(f"{split}: invalid batch shape {tuple(batch_images.shape)}")
        if len(batch_targets) != args.batch_size:
            split_errors.append(f"{split}: invalid target batch length")
        all_errors.extend(split_errors)
        print(f"\n{split.upper()}")
        print(f"  dataset images : {len(dataset)}")
        print(f"  checked images : {count}")
        print(f"  checked boxes  : {total_boxes}")
        print(f"  batch shape    : {tuple(batch_images.shape)}")
        print(f"  errors         : {len(split_errors)}")
        if split == "train":
            preview_samples.extend(dataset[index] for index in indices[:12])

    preview_path = reports_dir / "dataloader_preview_train.jpg"
    save_preview(preview_samples, preview_path)
    error_path = reports_dir / "dataloader_errors.txt"
    error_path.write_text("\n".join(all_errors) + ("\n" if all_errors else ""), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"STATUS       : {'PASS' if not all_errors else 'FAIL'}")
    print(f"TOTAL ERRORS : {len(all_errors)}")
    print(f"PREVIEW      : {preview_path}")
    print("=" * 72)
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
