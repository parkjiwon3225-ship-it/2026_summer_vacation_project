from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a tiny fixed sample to test training.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from rc_detector import (
        DetectionLoss,
        PersonDetectionDataset,
        PersonDetector,
        detection_collate,
    )

    seed = 20260811
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(args.device)
    dataset = PersonDetectionDataset(
        root / "data" / "processed" / "v1_grouped",
        "train",
        augment=False,
    )
    candidate_indices = np.linspace(0, len(dataset) - 1, args.images * 3, dtype=int)
    selected_indices: list[int] = []
    for index in candidate_indices:
        _, target = dataset[int(index)]
        if 1 <= len(target["boxes"]) <= 12:
            selected_indices.append(int(index))
        if len(selected_indices) == args.images:
            break
    if len(selected_indices) < args.images:
        raise RuntimeError(f"Could select only {len(selected_indices)} suitable images")
    subset = Subset(dataset, selected_indices)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        subset,
        batch_size=min(args.batch_size, len(subset)),
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
        generator=generator,
        drop_last=False,
    )
    model = PersonDetector().to(device)
    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    output_dir = root / "results" / "mini_overfit"
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    iterator = iter(loader)
    start_time = time.perf_counter()
    model.train()

    print("=" * 72)
    print("MINI-OVERFIT TRAINING TEST")
    print("=" * 72)
    print(f"Device       : {device}")
    print(f"Images       : {len(subset)}")
    print(f"Batch size   : {min(args.batch_size, len(subset))}")
    print(f"Iterations   : {args.iterations}")
    print(f"Learning rate: {args.learning_rate}")

    for iteration in range(1, args.iterations + 1):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        losses = criterion(predictions, targets)
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        row = {
            "iteration": iteration,
            "total": float(losses["total"].detach()),
            "classification": float(losses["classification"].detach()),
            "quality": float(losses["quality"].detach()),
            "box": float(losses["box"].detach()),
            "positive_count": int(losses["positive_count"].detach()),
            "gradient_norm": float(gradient_norm),
        }
        history.append(row)
        if iteration == 1 or iteration % max(args.iterations // 10, 1) == 0:
            print(
                f"iter {iteration:>4}/{args.iterations} | "
                f"total {row['total']:.4f} | cls {row['classification']:.4f} | "
                f"quality {row['quality']:.4f} | box {row['box']:.4f}"
            )

    elapsed = time.perf_counter() - start_time
    window = min(10, len(history))
    initial_loss = sum(row["total"] for row in history[:window]) / window
    final_loss = sum(row["total"] for row in history[-window:]) / window
    reduction = (initial_loss - final_loss) / initial_loss
    finite = all(np.isfinite(row["total"]) for row in history)
    passed = finite and reduction >= 0.50
    history_path = output_dir / "loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint_path = output_dir / "mini_overfit_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iterations": args.iterations,
            "selected_indices": selected_indices,
            "history": history,
        },
        checkpoint_path,
    )

    print("-" * 72)
    print(f"Initial mean loss : {initial_loss:.6f}")
    print(f"Final mean loss   : {final_loss:.6f}")
    print(f"Loss reduction    : {reduction:.2%}")
    print(f"Elapsed           : {elapsed:.2f} seconds")
    print(f"History           : {history_path}")
    print(f"Checkpoint        : {checkpoint_path}")
    print(f"STATUS            : {'PASS' if passed else 'FAIL'}")
    print("=" * 72)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
