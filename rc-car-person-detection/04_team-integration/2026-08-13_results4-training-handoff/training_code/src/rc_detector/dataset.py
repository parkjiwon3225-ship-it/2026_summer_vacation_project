from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PersonDetectionDataset(Dataset):
    """Read V1 YOLO labels and return letterboxed tensors with pixel xyxy boxes."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        image_size: tuple[int, int] = (320, 240),
        augment: bool = False,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.output_width, self.output_height = image_size
        self.augment = augment
        self.horizontal_flip_probability = horizontal_flip_probability
        self.images_dir = self.dataset_dir / split / "images"
        self.labels_dir = self.dataset_dir / split / "labels"
        if not self.images_dir.is_dir() or not self.labels_dir.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folders: {self.images_dir} or {self.labels_dir}"
            )
        self.image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")
        for image_path in self.image_paths:
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _read_normalized_labels(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
        boxes: list[list[float]] = []
        labels: list[int] = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid label at {label_path}:{line_number}")
            class_id = int(parts[0])
            if class_id != 0:
                raise ValueError(f"Expected person class 0 at {label_path}:{line_number}")
            center_x, center_y, width, height = map(float, parts[1:])
            boxes.append([center_x, center_y, width, height])
            labels.append(class_id)
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(labels, dtype=np.int64),
        )

    @staticmethod
    def _normalized_cxcywh_to_xyxy(
        boxes: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        result = np.empty_like(boxes)
        result[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
        result[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
        result[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
        result[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
        return result

    def _letterbox(
        self, image: Image.Image, boxes: np.ndarray
    ) -> tuple[Image.Image, np.ndarray, float, tuple[int, int]]:
        width, height = image.size
        scale = min(self.output_width / width, self.output_height / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        pad_x = (self.output_width - resized_width) // 2
        pad_y = (self.output_height - resized_height) // 2
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.output_width, self.output_height), (114, 114, 114))
        canvas.paste(resized, (pad_x, pad_y))
        transformed = boxes.copy()
        transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + pad_x
        transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + pad_y
        transformed[:, [0, 2]] = np.clip(transformed[:, [0, 2]], 0, self.output_width)
        transformed[:, [1, 3]] = np.clip(transformed[:, [1, 3]], 0, self.output_height)
        return canvas, transformed, scale, (pad_x, pad_y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        original_width, original_height = image.size
        normalized_boxes, labels = self._read_normalized_labels(label_path)
        boxes = self._normalized_cxcywh_to_xyxy(
            normalized_boxes, original_width, original_height
        )
        flipped = False
        if self.augment and random.random() < self.horizontal_flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            old_left = boxes[:, 0].copy()
            old_right = boxes[:, 2].copy()
            boxes[:, 0] = original_width - old_right
            boxes[:, 2] = original_width - old_left
            flipped = True
        image, boxes, scale, padding = self._letterbox(image, boxes)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        target: dict[str, Any] = {
            "boxes": torch.from_numpy(boxes).to(torch.float32),
            "labels": torch.from_numpy(labels).to(torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "original_size": torch.tensor(
                [original_height, original_width], dtype=torch.int64
            ),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "padding": torch.tensor(padding, dtype=torch.int64),
            "flipped": flipped,
            "path": str(image_path),
        }
        return image_tensor, target


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
