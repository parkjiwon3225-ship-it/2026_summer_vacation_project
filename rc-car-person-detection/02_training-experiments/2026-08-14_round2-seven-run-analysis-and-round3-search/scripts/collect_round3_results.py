from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path


PREFIX = "r3_school"
FILES = (
    "config.json",
    "device.json",
    "history.csv",
    "warnings.log",
    "runner_console.log",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parent
    source_root = root / "results" / "training"
    experiments = [path for path in sorted(source_root.glob(f"{PREFIX}*")) if path.is_dir()]
    experiments = [path for path in experiments if (path / "history.csv").is_file()]
    if not experiments:
        print("No Round 3 result with history.csv found.")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / f"ROUND3_RESULTS_TO_COPY_{stamp}"
    output.mkdir(parents=False, exist_ok=False)
    manifest_lines = []
    for experiment in experiments:
        destination = output / experiment.name
        destination.mkdir()
        for name in FILES:
            source = experiment / name
            if source.is_file():
                shutil.copy2(source, destination / name)
        checkpoint_destination = destination / "checkpoints"
        checkpoint_destination.mkdir()
        for name in ("best.pt", "last.pt"):
            source = experiment / "checkpoints" / name
            if source.is_file():
                shutil.copy2(source, checkpoint_destination / name)

    for path in sorted(output.rglob("*")):
        if path.is_file():
            relative = path.relative_to(output)
            manifest_lines.append(f"{sha256(path)}  {path.stat().st_size:>10}  {relative}")
    (output / "MANIFEST_SHA256.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    archive = Path(shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name))
    print("Collected experiments:", len(experiments))
    print("Folder:", output)
    print("ZIP   :", archive)
    print("Copy the ZIP to USB. The original training result remains unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
