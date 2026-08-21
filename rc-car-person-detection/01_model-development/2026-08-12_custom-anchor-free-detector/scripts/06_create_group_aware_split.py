from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPLITS = ("train", "valid", "test")
TARGET_RATIOS = {"train": 0.80, "valid": 0.10, "test": 0.10}
FEATURE_WEIGHTS = {
    "images": 5.0,
    "boxes": 2.0,
    "small16": 3.0,
    "small32": 3.0,
    "large96": 2.0,
    "empty_images": 1.0,
}


@dataclass
class Record:
    source_split: str
    image_path: Path
    group_id: str
    output_stem: str
    yolo_lines: list[str]
    boxes: int
    small16: int
    small32: int
    large96: int


@dataclass
class Group:
    group_id: str
    records: list[Record] = field(default_factory=list)
    features: dict[str, int] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a leakage-safe, group-aware YOLO train/valid/test split."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Read-only original image directory containing train/valid/test.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory. The source directory is never changed.",
    )
    return parser.parse_args()


def base_name(filename: str) -> str:
    name = Path(filename).stem
    return name.split(".rf.", 1)[0] if ".rf." in name else name


def source_group(name: str) -> str:
    patterns = (
        r"(.+?)[_-]frame[_-]?\d+.*$",
        r"(.+?)[_-]\d{4,6}$",
    )
    for pattern in patterns:
        match = re.match(pattern, name, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return name


def person_height_after_letterbox(
    width: float, height: float, box_height: float
) -> float:
    scale = min(320.0 / width, 240.0 / height)
    return box_height * scale


def collect_records(source_dir: Path, archive_dir: Path) -> list[Record]:
    records: list[Record] = []
    seen_output_names: set[str] = set()
    for split in SOURCE_SPLITS:
        csv_path = source_dir / f"{split}_annotations.csv"
        images_dir = archive_dir / split
        if not csv_path.is_file() or not images_dir.is_dir():
            raise FileNotFoundError(
                f"Expected annotation CSV and archive images: {csv_path} and {images_dir}"
            )
        rows_by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"filename", "width", "height", "xmin", "ymin", "xmax", "ymax"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Missing required columns in {csv_path}")
            for row in reader:
                rows_by_filename[row["filename"]].append(row)
        for filename in sorted(rows_by_filename):
            image_path = images_dir / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in CSV was not found: {image_path}")
            rows = rows_by_filename[filename]
            width = float(rows[0]["width"])
            height = float(rows[0]["height"])
            heights = [
                person_height_after_letterbox(
                    width, height, float(row["ymax"]) - float(row["ymin"])
                )
                for row in rows
            ]
            yolo_lines: list[str] = []
            for row in rows:
                xmin, ymin = float(row["xmin"]), float(row["ymin"])
                xmax, ymax = float(row["xmax"]), float(row["ymax"])
                x_center = ((xmin + xmax) / 2.0) / width
                y_center = ((ymin + ymax) / 2.0) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                yolo_lines.append(
                    f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
                )
            clean_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem)
            output_stem = f"{split}__{clean_stem}"
            if output_stem in seen_output_names:
                raise ValueError(f"Duplicate output name: {output_stem}")
            seen_output_names.add(output_stem)
            group_id = source_group(base_name(image_path.name)).lower()
            records.append(
                Record(
                    source_split=split,
                    image_path=image_path,
                    group_id=group_id,
                    output_stem=output_stem,
                    yolo_lines=yolo_lines,
                    boxes=len(rows),
                    small16=sum(height < 16 for height in heights),
                    small32=sum(height < 32 for height in heights),
                    large96=sum(height >= 96 for height in heights),
                )
            )
    return records


def make_groups(records: list[Record]) -> list[Group]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    groups: list[Group] = []
    for group_id, group_records in grouped.items():
        features = {
            "images": len(group_records),
            "boxes": sum(record.boxes for record in group_records),
            "small16": sum(record.small16 for record in group_records),
            "small32": sum(record.small32 for record in group_records),
            "large96": sum(record.large96 for record in group_records),
            "empty_images": sum(record.boxes == 0 for record in group_records),
        }
        groups.append(Group(group_id, group_records, features))
    return groups


def totals_for(groups: list[Group]) -> dict[str, int]:
    return {key: sum(group.features[key] for group in groups) for key in FEATURE_WEIGHTS}


def score_assignment(
    split_totals: dict[str, dict[str, int]], totals: dict[str, int]
) -> float:
    score = 0.0
    for split, ratio in TARGET_RATIOS.items():
        for feature, weight in FEATURE_WEIGHTS.items():
            target = totals[feature] * ratio
            denominator = max(target, 1.0)
            error = (split_totals[split][feature] - target) / denominator
            score += weight * error * error
    return score


def split_groups(groups: list[Group], seed: int, trials: int) -> dict[str, str]:
    totals = totals_for(groups)
    best_score = float("inf")
    best_assignment: dict[str, str] | None = None
    for trial in range(trials):
        rng = random.Random(seed + trial)
        ordered = list(groups)
        rng.shuffle(ordered)
        ordered.sort(
            key=lambda group: (
                group.features["images"],
                group.features["boxes"],
                group.features["small32"],
            ),
            reverse=True,
        )
        split_totals = {
            split: {feature: 0 for feature in FEATURE_WEIGHTS} for split in TARGET_RATIOS
        }
        assignment: dict[str, str] = {}
        for group in ordered:
            choices: list[tuple[float, float, str]] = []
            for split in TARGET_RATIOS:
                for feature in FEATURE_WEIGHTS:
                    split_totals[split][feature] += group.features[feature]
                candidate_score = score_assignment(split_totals, totals)
                image_fill = split_totals[split]["images"] / max(
                    totals["images"] * TARGET_RATIOS[split], 1.0
                )
                choices.append((candidate_score, image_fill, split))
                for feature in FEATURE_WEIGHTS:
                    split_totals[split][feature] -= group.features[feature]
            _, _, selected = min(choices)
            assignment[group.group_id] = selected
            for feature in FEATURE_WEIGHTS:
                split_totals[selected][feature] += group.features[feature]
        final_score = score_assignment(split_totals, totals)
        if final_score < best_score:
            best_score = final_score
            best_assignment = assignment
    if best_assignment is None:
        raise RuntimeError("Could not create a split assignment")
    return best_assignment


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}\n"
                "Use --overwrite only if you intentionally want to rebuild V1."
            )
        shutil.rmtree(output_dir)
    for split in TARGET_RATIOS:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)


def materialize(
    records: list[Record], assignment: dict[str, str], output_dir: Path
) -> None:
    manifest_rows: list[dict[str, object]] = []
    for record in records:
        destination_split = assignment[record.group_id]
        image_target = (
            output_dir
            / destination_split
            / "images"
            / f"{record.output_stem}{record.image_path.suffix.lower()}"
        )
        label_target = output_dir / destination_split / "labels" / f"{record.output_stem}.txt"
        shutil.copy2(record.image_path, image_target)
        label_target.write_text("\n".join(record.yolo_lines) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "new_split": destination_split,
                "group_id": record.group_id,
                "source_split": record.source_split,
                "source_image": str(record.image_path),
                "output_image": str(image_target),
                "boxes": record.boxes,
                "small16": record.small16,
                "small32": record.small32,
                "large96": record.large96,
            }
        )
    manifest_path = output_dir / "reports" / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)


def verify_and_report(
    records: list[Record], assignment: dict[str, str], output_dir: Path, seed: int
) -> dict[str, object]:
    groups_by_split = {
        split: {group for group, assigned in assignment.items() if assigned == split}
        for split in TARGET_RATIOS
    }
    leakage = (
        (groups_by_split["train"] & groups_by_split["valid"])
        | (groups_by_split["train"] & groups_by_split["test"])
        | (groups_by_split["valid"] & groups_by_split["test"])
    )
    if leakage:
        raise RuntimeError(f"Group leakage detected: {sorted(leakage)[:10]}")
    report: dict[str, object] = {
        "seed": seed,
        "target_ratios": TARGET_RATIOS,
        "group_leakage_count": len(leakage),
        "splits": {},
    }
    for split in TARGET_RATIOS:
        selected = [record for record in records if assignment[record.group_id] == split]
        report["splits"][split] = {
            "images": len(selected),
            "groups": len(groups_by_split[split]),
            "boxes": sum(record.boxes for record in selected),
            "small16": sum(record.small16 for record in selected),
            "small32": sum(record.small32 for record in selected),
            "large96": sum(record.large96 for record in selected),
        }
    (output_dir / "reports" / "split_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: person\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    return report


def print_report(report: dict[str, object], output_dir: Path) -> None:
    print("=" * 72)
    print("GROUP-AWARE SPLIT V1")
    print("=" * 72)
    for split, stats in report["splits"].items():
        print(f"\n{split.upper()}")
        for key, value in stats.items():
            print(f"  {key:<12}: {value}")
    print(f"\nGroup leakage : {report['group_leakage_count']}")
    print(f"Saved to      : {output_dir}")
    print(f"Dataset YAML  : {output_dir / 'dataset.yaml'}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_dir = (args.source or root / "data" / "processed" / "v0").resolve()
    archive_dir = (args.archive or root / "archive").resolve()
    output_dir = (args.output or root / "data" / "processed" / "v1_grouped").resolve()
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("Output must not be the source directory or a child of it")
    print(f"Reading annotations (read-only): {source_dir}")
    print(f"Reading archive images (read-only): {archive_dir}")
    records = collect_records(source_dir, archive_dir)
    groups = make_groups(records)
    print(f"Found {len(records)} images in {len(groups)} source groups")
    assignment = split_groups(groups, args.seed, args.trials)
    prepare_output(output_dir, args.overwrite)
    materialize(records, assignment, output_dir)
    report = verify_and_report(records, assignment, output_dir, args.seed)
    print_report(report, output_dir)


if __name__ == "__main__":
    main()
