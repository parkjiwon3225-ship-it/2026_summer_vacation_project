from __future__ import annotations

import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PersonDetectionDataset, detection_collate
from .assignment import AnchorFreeTargetAssigner
from .losses import DetectionLoss
from .inference import DetectionPostProcessor
from .metrics import DetectionEvaluator
from .model import PersonDetector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    worker_seed_value = torch.initial_seed() % (2**32)
    random.seed(worker_seed_value)
    np.random.seed(worker_seed_value)


def describe_device(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_vram_gib": round(properties.total_memory / (1024**3), 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return result


def create_loader(
    dataset: PersonDetectionDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=detection_collate,
        worker_init_fn=worker_seed,
        generator=generator,
        drop_last=shuffle,
    )


def autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def create_grad_scaler(
    enabled: bool,
    initial_scale: float = 1024.0,
    growth_interval: int = 4000,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
            init_scale=initial_scale,
            growth_interval=growth_interval,
        )
    except TypeError:
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
            init_scale=initial_scale,
            growth_interval=growth_interval,
        )


def move_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    return images.to(device, non_blocking=device.type == "cuda")


def train_one_epoch(
    model: PersonDetector,
    criterion: DetectionLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    accumulation_steps: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    sums = {"total": 0.0, "classification": 0.0, "quality": 0.0, "box": 0.0}
    positive_sum = 0.0
    gradient_norm_sum = 0.0
    update_count = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader, 1):
        images = move_images(images, device)
        with autocast_context(device, amp_enabled):
            predictions = model(images)
            losses = criterion(predictions, targets)
            scaled_loss = losses["total"] / accumulation_steps
        scaler.scale(scaled_loss).backward()
        should_update = step % accumulation_steps == 0 or step == len(loader)
        if should_update:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            gradient_norm_sum += float(gradient_norm)
            update_count += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for key in sums:
            sums[key] += float(losses[key].detach())
        positive_sum += float(losses["positive_count"].detach())
    batch_count = max(len(loader), 1)
    return {
        **{key: value / batch_count for key, value in sums.items()},
        "positive_count": positive_sum / batch_count,
        "gradient_norm": gradient_norm_sum / max(update_count, 1),
    }


@torch.inference_mode()
def validate_one_epoch(
    model: PersonDetector,
    criterion: DetectionLoss,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "classification": 0.0, "quality": 0.0, "box": 0.0}
    positive_sum = 0.0
    for images, targets in loader:
        images = move_images(images, device)
        with autocast_context(device, amp_enabled):
            predictions = model(images)
            losses = criterion(predictions, targets)
        for key in sums:
            sums[key] += float(losses[key])
        positive_sum += float(losses["positive_count"])
    batch_count = max(len(loader), 1)
    return {
        **{key: value / batch_count for key, value in sums.items()},
        "positive_count": positive_sum / batch_count,
    }


@torch.inference_mode()
def evaluate_detections(
    model: PersonDetector,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    postprocessor: DetectionPostProcessor,
    operating_score_threshold: float,
    image_size: tuple[int, int] = (320, 240),
) -> dict[str, float | int]:
    model.eval()
    evaluator = DetectionEvaluator(operating_score_threshold)
    for images, targets in loader:
        images = move_images(images, device)
        with autocast_context(device, amp_enabled):
            predictions = model(images)
        detections = postprocessor(predictions, image_size=image_size)
        evaluator.update(detections, targets)
    metrics = evaluator.compute()
    flattened: dict[str, float | int] = {}
    for key, value in metrics.items():
        if key == "size_recall":
            for size_name, size_values in value.items():
                for size_key, size_value in size_values.items():
                    flattened[f"{size_name}_{size_key}"] = size_value
        elif isinstance(value, (int, float)):
            flattened[key] = value
    return flattened


def health_warnings(
    epoch: int,
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
    detection_metrics: dict[str, float | int] | None,
) -> list[str]:
    warnings: list[str] = []
    numeric_values = list(train_metrics.values()) + list(valid_metrics.values())
    if not all(np.isfinite(value) for value in numeric_values):
        warnings.append("NaN or Inf detected in loss/optimization metrics")
    if train_metrics.get("gradient_norm", 0.0) > 100:
        warnings.append("Gradient norm exceeds 100")
    if epoch >= 5 and valid_metrics["total"] > train_metrics["total"] * 2.0:
        warnings.append("Validation loss is more than twice the training loss")
    if detection_metrics is not None and epoch >= 5:
        if float(detection_metrics.get("no_detection_image_rate", 0.0)) > 0.95:
            warnings.append("More than 95% of validation images have no detections")
        if float(detection_metrics.get("average_detections_per_image", 0.0)) > 50:
            warnings.append("Average detections per image exceeds 50")
        if epoch >= 10 and float(detection_metrics.get("recall", 0.0)) < 0.05:
            warnings.append("Validation recall remains below 5%")
    return warnings


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_checkpoint(
    path: Path,
    model: PersonDetector,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> tuple[int, float, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_map50_95", -1.0)),
        float(checkpoint.get("best_validation_loss", float("inf"))),
    )


def load_model_weights(
    path: Path,
    model: PersonDetector,
    device: torch.device,
) -> dict[str, Any]:
    """Load model parameters only and intentionally reset training state.

    Resolution curricula must not restore the previous stage's optimizer,
    scheduler, scaler, epoch, or best score.  The convolutional detector has
    resolution-independent parameter shapes, so the state dict can be reused
    when moving from 640/480 inputs back to the final 320x240 input.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        metadata = {
            "source_epoch": int(checkpoint.get("epoch", -1)),
            "source_experiment": str(
                checkpoint.get("config", {}).get("experiment_name", "unknown")
            ),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {"source_epoch": -1, "source_experiment": "raw_state_dict"}
    else:
        raise TypeError(f"Unsupported weights payload in {path}")
    model.load_state_dict(state_dict, strict=True)
    return metadata


def run_training(
    root: Path,
    config: dict[str, Any],
    resume_path: Path | None = None,
    initial_weights_path: Path | None = None,
) -> dict[str, Any]:
    if resume_path is not None and initial_weights_path is not None:
        raise ValueError("resume_path and initial_weights_path are mutually exclusive")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    dataset_dir = root / "data" / "processed" / "v1_grouped"
    image_size = (int(config["image_width"]), int(config["image_height"]))
    train_dataset = PersonDetectionDataset(
        dataset_dir,
        "train",
        image_size=image_size,
        augment=True,
        horizontal_flip_probability=float(config["horizontal_flip_probability"]),
    )
    valid_dataset = PersonDetectionDataset(
        dataset_dir,
        "valid",
        image_size=image_size,
        augment=False,
    )
    train_loader = create_loader(
        train_dataset,
        int(config["batch_size"]),
        int(config["num_workers"]),
        True,
        int(config["seed"]),
        device,
    )
    valid_loader = create_loader(
        valid_dataset,
        int(config["batch_size"]),
        int(config["num_workers"]),
        False,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"])
    )
    scaler = create_grad_scaler(
        amp_enabled,
        initial_scale=float(config["amp_initial_scale"]),
        growth_interval=int(config["amp_growth_interval"]),
    )
    experiment_dir = root / "results" / "training" / str(config["experiment_name"])
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    device_info = describe_device(device)
    (experiment_dir / "device.json").write_text(
        json.dumps(device_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_path = experiment_dir / "history.csv"
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(experiment_dir / "tensorboard"))
    except ImportError:
        print("TensorBoard is unavailable; CSV history will still be recorded.")
    postprocessor = DetectionPostProcessor(
        score_threshold=float(config["evaluation_score_floor"]),
        nms_iou_threshold=float(config["nms_iou_threshold"]),
        max_detections=int(config["max_detections"]),
    )
    start_epoch, best_map50_95, best_validation_loss = 1, -1.0, float("inf")
    initialization: dict[str, Any] = {"mode": "fresh"}
    if resume_path is not None:
        start_epoch, best_map50_95, best_validation_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        initialization = {"mode": "resume", "path": str(resume_path)}
    elif initial_weights_path is not None:
        metadata = load_model_weights(initial_weights_path, model, device)
        initialization = {
            "mode": "weights_only",
            "path": str(initial_weights_path),
            **metadata,
        }

    max_runtime_hours = float(config.get("max_runtime_hours", 0.0) or 0.0)
    runtime_limit_seconds = (
        max_runtime_hours * 3600.0 if max_runtime_hours > 0.0 else None
    )
    training_started = time.perf_counter()

    print("=" * 72)
    print("FULL TRAINING")
    print("=" * 72)
    for key, value in device_info.items():
        print(f"{key:<20}: {value}")
    print(f"train_images        : {len(train_dataset)}")
    print(f"valid_images        : {len(valid_dataset)}")
    print(f"batch_size          : {config['batch_size']}")
    print(f"image_size          : {image_size[0]}x{image_size[1]}")
    print(f"amp_enabled         : {amp_enabled}")
    print(f"start_epoch         : {start_epoch}")
    print(f"initialization      : {initialization['mode']}")
    print(
        f"max_runtime_hours   : {max_runtime_hours:.3f}"
        if runtime_limit_seconds is not None
        else "max_runtime_hours   : unlimited"
    )

    stop_reason = "epochs_complete"
    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            int(config["gradient_accumulation"]),
            float(config["gradient_clip_norm"]),
        )
        train_seconds = time.perf_counter() - train_start
        valid_metrics = validate_one_epoch(
            model, criterion, valid_loader, device, amp_enabled
        )
        detection_metrics = None
        if epoch % int(config["metrics_every"]) == 0 or epoch == int(config["epochs"]):
            detection_metrics = evaluate_detections(
                model,
                valid_loader,
                device,
                amp_enabled,
                postprocessor,
                float(config["operating_score_threshold"]),
                image_size,
            )
        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        peak_vram_gib = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            **(
                {f"metric_{key}": value for key, value in detection_metrics.items()}
                if detection_metrics is not None
                else {}
            ),
            "train_images_per_second": len(train_dataset) / max(train_seconds, 1e-7),
            "peak_vram_gib": peak_vram_gib,
            "amp_scale": float(scaler.get_scale()),
            "seconds": elapsed,
        }
        warnings = health_warnings(epoch, train_metrics, valid_metrics, detection_metrics)
        row["warning_count"] = len(warnings)
        append_history(history_path, row)
        if warnings:
            with (experiment_dir / "warnings.log").open("a", encoding="utf-8") as handle:
                for warning in warnings:
                    handle.write(f"epoch {epoch}: {warning}\n")
        if writer is not None:
            writer.add_scalars(
                "loss/total",
                {"train": train_metrics["total"], "valid": valid_metrics["total"]},
                epoch,
            )
            for key in ("classification", "quality", "box"):
                writer.add_scalars(
                    f"loss/{key}",
                    {"train": train_metrics[key], "valid": valid_metrics[key]},
                    epoch,
                )
            writer.add_scalar("optimization/learning_rate", row["learning_rate"], epoch)
            writer.add_scalar("optimization/gradient_norm", train_metrics["gradient_norm"], epoch)
            writer.add_scalar("performance/images_per_second", row["train_images_per_second"], epoch)
            writer.add_scalar("performance/peak_vram_gib", peak_vram_gib, epoch)
            if detection_metrics is not None:
                for key, value in detection_metrics.items():
                    writer.add_scalar(f"detection/{key}", value, epoch)
            writer.flush()
        if valid_metrics["total"] < best_validation_loss:
            best_validation_loss = valid_metrics["total"]
        current_map = (
            float(detection_metrics["map50_95"])
            if detection_metrics is not None
            else -1.0
        )
        is_best = current_map > best_map50_95
        if is_best:
            best_map50_95 = current_map
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "best_map50_95": best_map50_95,
            "config": config,
            "device": device_info,
        }
        if epoch % int(config["checkpoint_every"]) == 0:
            atomic_torch_save(payload, checkpoint_dir / "last.pt")
        if is_best:
            atomic_torch_save(payload, checkpoint_dir / "best.pt")
        print(
            f"epoch {epoch:>3}/{config['epochs']} | "
            f"train {train_metrics['total']:.4f} | "
            f"valid {valid_metrics['total']:.4f} | "
            f"mAP {current_map:.4f} | best {best_map50_95:.4f} | {elapsed:.1f}s"
        )
        for warning in warnings:
            print(f"WARNING: {warning}")
        last_completed_epoch = epoch
        if (
            runtime_limit_seconds is not None
            and time.perf_counter() - training_started >= runtime_limit_seconds
        ):
            if not (checkpoint_dir / "last.pt").is_file() or (
                epoch % int(config["checkpoint_every"]) != 0
            ):
                atomic_torch_save(payload, checkpoint_dir / "last.pt")
            stop_reason = "runtime_limit"
            print(
                f"TIME BUDGET REACHED after epoch {epoch}; "
                "the last completed checkpoint is safe."
            )
            break
    if writer is not None:
        writer.close()
    status = {
        "experiment_name": str(config["experiment_name"]),
        "stop_reason": stop_reason,
        "last_completed_epoch": int(last_completed_epoch),
        "requested_epochs": int(config["epochs"]),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "elapsed_hours": (time.perf_counter() - training_started) / 3600.0,
        "best_map50_95": float(best_map50_95),
        "initialization": initialization,
    }
    (experiment_dir / "training_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status
