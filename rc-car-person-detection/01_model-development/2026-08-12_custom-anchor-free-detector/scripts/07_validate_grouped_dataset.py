from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate grouped YOLO dataset V1.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def image_map(folder: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem: {path.stem}")
            result[path.stem] = path
    return result


def label_map(folder: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(folder.glob("*.txt"))}


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected 5 values, found {len(parts)}")
            continue
        try:
            class_float, x, y, width, height = (float(value) for value in parts)
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric value")
            continue
        class_id = int(class_float)
        if class_float != class_id or class_id != 0:
            errors.append(f"{path}:{line_number}: class must be integer 0")
        values = (x, y, width, height)
        if not all(value == value and abs(value) != float("inf") for value in values):
            errors.append(f"{path}:{line_number}: NaN or infinite coordinate")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"{path}:{line_number}: box width/height must be positive")
        if not (-TOLERANCE <= x <= 1 + TOLERANCE and -TOLERANCE <= y <= 1 + TOLERANCE):
            errors.append(f"{path}:{line_number}: center outside normalized range")
        if x - width / 2 < -TOLERANCE or x + width / 2 > 1 + TOLERANCE:
            errors.append(f"{path}:{line_number}: box exceeds image horizontally")
        if y - height / 2 < -TOLERANCE or y + height / 2 > 1 + TOLERANCE:
            errors.append(f"{path}:{line_number}: box exceeds image vertically")
        boxes.append((class_id, x, y, width, height))
    return boxes, errors


def verify_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                return f"{path}: invalid image dimensions"
    except Exception as error:
        return f"{path}: image decode failed: {error}"
    return None


def validate_manifest(dataset_dir: Path) -> tuple[dict[str, int], list[str]]:
    path = dataset_dir / "reports" / "split_manifest.csv"
    errors: list[str] = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    row_counts = {split: 0 for split in SPLITS}
    if not path.is_file():
        return row_counts, [f"Missing manifest: {path}"]
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"new_split", "group_id", "output_image"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            return row_counts, [f"Manifest is missing required columns: {path}"]
        for row_number, row in enumerate(reader, 2):
            split = row["new_split"]
            if split not in SPLITS:
                errors.append(f"Manifest row {row_number}: invalid split {split!r}")
                continue
            row_counts[split] += 1
            group_splits[row["group_id"]].add(split)
            if not Path(row["output_image"]).is_file():
                errors.append(f"Manifest row {row_number}: output image missing")
    leaking = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    for group, splits in list(leaking.items())[:20]:
        errors.append(f"Group leakage: {group} appears in {sorted(splits)}")
    return row_counts, errors


def make_preview(
    split: str,
    images: dict[str, Path],
    labels: dict[str, Path],
    output_path: Path,
    sample_count: int,
    seed: int,
) -> None:
    candidates = sorted(images.keys() & labels.keys())
    rng = random.Random(seed + SPLITS.index(split))
    selected = rng.sample(candidates, min(sample_count, len(candidates)))
    tile_width, tile_height = 320, 260
    columns = 4
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202124")
    font = ImageFont.load_default()
    for index, stem in enumerate(selected):
        with Image.open(images[stem]) as source:
            image = source.convert("RGB")
        image.thumbnail((tile_width, tile_height - 20))
        tile = Image.new("RGB", (tile_width, tile_height - 20), "black")
        offset_x = (tile_width - image.width) // 2
        offset_y = (tile_height - 20 - image.height) // 2
        tile.paste(image, (offset_x, offset_y))
        draw = ImageDraw.Draw(tile)
        boxes, _ = parse_label(labels[stem])
        for _, x, y, width, height in boxes:
            left = offset_x + (x - width / 2) * image.width
            top = offset_y + (y - height / 2) * image.height
            right = offset_x + (x + width / 2) * image.width
            bottom = offset_y + (y + height / 2) * image.height
            draw.rectangle((left, top, right, bottom), outline="#00ff66", width=2)
        column, row = index % columns, index // columns
        canvas.paste(tile, (column * tile_width, row * tile_height))
        canvas_draw = ImageDraw.Draw(canvas)
        short_name = stem if len(stem) <= 42 else stem[:39] + "..."
        canvas_draw.text(
            (column * tile_width + 4, row * tile_height + tile_height - 17),
            f"{short_name} | boxes={len(boxes)}",
            fill="white",
            font=font,
        )
    canvas.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_dir = (args.dataset or root / "data" / "processed" / "v1_grouped").resolve()
    reports_dir = dataset_dir / "reports"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_errors: list[str] = []
    report: dict[str, object] = {"dataset": str(dataset_dir), "splits": {}}
    maps: dict[str, tuple[dict[str, Path], dict[str, Path]]] = {}
    print("=" * 72)
    print("GROUPED DATASET V1 INTEGRITY CHECK")
    print("=" * 72)

    for split in SPLITS:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing split folders: {images_dir} or {labels_dir}")
        images = image_map(images_dir)
        labels = label_map(labels_dir)
        maps[split] = (images, labels)
        missing_labels = sorted(images.keys() - labels.keys())
        orphan_labels = sorted(labels.keys() - images.keys())
        split_errors = [f"Missing label: {stem}" for stem in missing_labels]
        split_errors.extend(f"Orphan label: {stem}" for stem in orphan_labels)
        box_count = 0
        empty_labels = 0
        for stem in sorted(images.keys() & labels.keys()):
            image_error = verify_image(images[stem])
            if image_error:
                split_errors.append(image_error)
            boxes, label_errors = parse_label(labels[stem])
            box_count += len(boxes)
            empty_labels += not boxes
            split_errors.extend(label_errors)
        all_errors.extend(f"[{split}] {error}" for error in split_errors)
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": box_count,
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "empty_labels": empty_labels,
            "errors": len(split_errors),
        }
        print(f"\n{split.upper()}")
        for key, value in report["splits"][split].items():
            print(f"  {key:<15}: {value}")

    manifest_counts, manifest_errors = validate_manifest(dataset_dir)
    all_errors.extend(f"[manifest] {error}" for error in manifest_errors)
    for split in SPLITS:
        image_count = report["splits"][split]["images"]
        if manifest_counts[split] != image_count:
            all_errors.append(
                f"[manifest] {split}: {manifest_counts[split]} rows but {image_count} images"
            )

    for split, (images, labels) in maps.items():
        make_preview(
            split,
            images,
            labels,
            reports_dir / f"bbox_preview_{split}.jpg",
            args.samples,
            args.seed,
        )

    report["manifest_rows"] = manifest_counts
    report["total_errors"] = len(all_errors)
    report["status"] = "PASS" if not all_errors else "FAIL"
    (reports_dir / "integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "integrity_errors.txt").write_text(
        "\n".join(all_errors) + ("\n" if all_errors else ""), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"STATUS          : {report['status']}")
    print(f"TOTAL ERRORS    : {len(all_errors)}")
    print(f"REPORT          : {reports_dir / 'integrity_report.json'}")
    print(f"BBOX PREVIEWS   : {reports_dir / 'bbox_preview_<split>.jpg'}")
    print("=" * 72)
    if all_errors:
        print(f"First errors are recorded in: {reports_dir / 'integrity_errors.txt'}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
