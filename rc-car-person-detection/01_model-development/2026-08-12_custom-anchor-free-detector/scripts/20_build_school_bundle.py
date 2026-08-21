from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORIES = (
    "src",
    "scripts",
    "configs",
    "notebooks",
    "docs",
    "data/processed/v1_grouped",
    ".vscode",
)
INCLUDE_FILES = (
    "environment-school.yml",
    "pyproject.toml",
    "00_학교노트북_설치순서.txt",
    "01_노트북별_실험배정.txt",
    "02_오류발생시_확인사항.txt",
    "00_verify_school_setup.bat",
)
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the school GPU training bundle.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--bundle-name",
        default=f"rc_person_detector_school_bundle_{date.today():%Y%m%d}",
        help="ZIP filename and top-level folder name (for example: RC)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Read-only path to v1_grouped when it is outside --root",
    )
    return parser.parse_args()


def included_files(root: Path, dataset_root: Path | None) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for directory in INCLUDE_DIRECTORIES:
        path = dataset_root if directory == "data/processed/v1_grouped" and dataset_root else root / directory
        if not path.is_dir():
            raise FileNotFoundError(f"Required directory missing: {path}")
        relative_base = Path(directory)
        files.extend(
            (candidate, relative_base / candidate.relative_to(path))
            for candidate in path.rglob("*")
            if candidate.is_file()
        )
    for filename in INCLUDE_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")
        files.append((path, Path(filename)))
    return sorted(
        (
            (path, relative)
            for path, relative in files
            if not EXCLUDED_PARTS.intersection(path.parts)
            and path.suffix.lower() not in EXCLUDED_SUFFIXES
        ),
        key=lambda item: item[1].as_posix(),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = args.bundle_name.strip()
    if not bundle_name or any(character in bundle_name for character in '<>:"/\\|?*'):
        raise ValueError(f"Invalid bundle name: {bundle_name!r}")
    zip_path = output_dir / f"{bundle_name}.zip"
    temporary_zip = output_dir / f"{bundle_name}.zip.tmp"
    manifest_path = output_dir / f"{bundle_name}_sha256_manifest.json"
    dataset_root = args.dataset_root.resolve() if args.dataset_root else None
    files = included_files(root, dataset_root)
    records: list[dict[str, object]] = []
    total_bytes = sum(path.stat().st_size for path, _ in files)

    print("=" * 72)
    print("BUILD SCHOOL GPU BUNDLE")
    print("=" * 72)
    print(f"Files       : {len(files):,}")
    print(f"Input size  : {total_bytes / (1024**3):.2f} GiB")
    print(f"Output      : {zip_path}")

    for index, (path, relative_path) in enumerate(files, 1):
        relative = relative_path.as_posix()
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        if index % 5000 == 0:
            print(f"Hashed      : {index:,}/{len(files):,}")
    manifest = {
        "bundle": bundle_name,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "excluded": ["archive", "data/processed/v0", "results", "checkpoints"],
        "files": records,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    with zipfile.ZipFile(
        temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for index, (path, relative_path) in enumerate(files, 1):
            relative = relative_path.as_posix()
            archive.write(path, f"{bundle_name}/{relative}")
            if index % 5000 == 0:
                print(f"Archived    : {index:,}/{len(files):,}")
        archive.writestr(
            f"{bundle_name}/SHA256_MANIFEST.json",
            manifest_text,
        )
    temporary_zip.replace(zip_path)
    zip_hash = sha256(zip_path)
    zip_hash_path = output_dir / f"{bundle_name}.zip.sha256.txt"
    zip_hash_path.write_text(f"{zip_hash}  {zip_path.name}\n", encoding="ascii")

    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_file = archive.testzip()
        archive_count = len(archive.infolist())
    if bad_file is not None:
        raise RuntimeError(f"ZIP CRC validation failed: {bad_file}")
    if archive_count != len(files) + 1:
        raise RuntimeError(
            f"ZIP entry count mismatch: expected {len(files) + 1}, found {archive_count}"
        )

    print("-" * 72)
    print(f"ZIP size    : {zip_path.stat().st_size / (1024**3):.2f} GiB")
    print(f"ZIP SHA-256 : {zip_hash}")
    print(f"Manifest    : {manifest_path}")
    print(f"CRC check   : PASS")
    print(f"STATUS      : PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
