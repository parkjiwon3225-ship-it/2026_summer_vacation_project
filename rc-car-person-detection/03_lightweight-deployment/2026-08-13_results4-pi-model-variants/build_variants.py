from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


WIDTH, HEIGHT = 320, 240
LEVELS = ("p2", "p3", "p4", "p5")
STRIDES = (4, 8, 16, 32)


class ExportWrapper(torch.nn.Module):
    """Export raw model predictions as concatenated pixel-space boxes and scores."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = self.model(image)
        all_boxes, all_scores = [], []
        for level, stride in zip(LEVELS, STRIDES, strict=True):
            output = predictions[level]
            cls = torch.sigmoid(output["class_logits"][:, 0])
            quality = torch.sigmoid(output["quality_logits"][:, 0])
            scores = (cls * quality).reshape(image.shape[0], -1)
            distances = output["distances"].permute(0, 2, 3, 1).reshape(image.shape[0], -1, 4)
            height, width = output["class_logits"].shape[-2:]
            x = (torch.arange(width, device=image.device, dtype=image.dtype) + 0.5) * stride
            y = (torch.arange(height, device=image.device, dtype=image.dtype) + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
            left_top = points[None] - distances[..., :2]
            right_bottom = points[None] + distances[..., 2:]
            boxes = torch.cat((left_top, right_bottom), dim=-1)
            all_boxes.append(boxes)
            all_scores.append(scores)
        return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1)


def letterbox(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(WIDTH / width, HEIGHT / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    pad_x = (WIDTH - resized_width) // 2
    pad_y = (HEIGHT - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


class Reader(CalibrationDataReader):
    def __init__(self, paths: list[Path], input_name: str) -> None:
        self.paths = paths
        self.input_name = input_name
        self.rewind()

    def get_next(self):
        try:
            path = next(self.iterator)
        except StopIteration:
            return None
        return {self.input_name: letterbox(path)}

    def rewind(self):
        self.iterator = iter(self.paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--calibration-images", type=Path, required=True)
    parser.add_argument("--verification-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=192)
    args = parser.parse_args()

    output = args.output.resolve()
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root.resolve()))
    from rc_detector.model import PersonDetector

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = PersonDetector(
        fpn_channels=int(config["fpn_channels"]),
        backbone_expansion=float(config["backbone_expansion"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    wrapped = ExportWrapper(model).eval()

    fp32_path = models_dir / "results4_fpn48_fp32.onnx"
    dummy = torch.zeros(1, 3, HEIGHT, WIDTH, dtype=torch.float32)
    with torch.inference_mode():
        reference = tuple(value.cpu().numpy() for value in wrapped(dummy))
        torch.onnx.export(
            wrapped,
            dummy,
            fp32_path,
            input_names=["images"],
            output_names=["boxes", "scores"],
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    image_paths = sorted(
        path for path in args.calibration_images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if len(image_paths) < args.calibration_count:
        raise RuntimeError("Not enough validation images for calibration")
    indices = np.linspace(0, len(image_paths) - 1, args.calibration_count, dtype=int)
    calibration_paths = [image_paths[index] for index in indices]

    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    variants = {
        "int8_qdq_minmax": (QuantFormat.QDQ, CalibrationMethod.MinMax),
        "int8_qdq_percentile": (QuantFormat.QDQ, CalibrationMethod.Percentile),
        "int8_qoperator_minmax": (QuantFormat.QOperator, CalibrationMethod.MinMax),
    }
    for name, (quant_format, method) in variants.items():
        reader = Reader(calibration_paths, input_name)
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(models_dir / f"results4_fpn48_{name}.onnx"),
            calibration_data_reader=reader,
            quant_format=quant_format,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            calibrate_method=method,
            op_types_to_quantize=["Conv"],
        )

    verification_paths = sorted(
        path for path in args.verification_images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    test_paths = verification_paths[:: max(1, len(verification_paths) // 16)][:16]
    report = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_best_map50_95": float(checkpoint["best_map50_95"]),
        "checkpoint_sha256": sha256(args.checkpoint),
        "calibration_images": len(calibration_paths),
        "calibration_source": "grouped train; evenly spaced deterministic sample",
        "verification_source": "grouped valid; 16 evenly spaced deterministic images",
        "models": [],
    }
    reference_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    reference_outputs = [reference_session.run(None, {input_name: letterbox(p)}) for p in test_paths]
    for path in sorted(models_dir.glob("*.onnx")):
        onnx.checker.check_model(onnx.load(path))
        candidate_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        box_errors, score_errors, score_maes = [], [], []
        for image_path, ref in zip(test_paths, reference_outputs, strict=True):
            candidate = candidate_session.run(None, {candidate_session.get_inputs()[0].name: letterbox(image_path)})
            box_errors.append(float(np.max(np.abs(ref[0] - candidate[0]))))
            score_errors.append(float(np.max(np.abs(ref[1] - candidate[1]))))
            score_maes.append(float(np.mean(np.abs(ref[1] - candidate[1]))))
        report["models"].append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "size_mib": path.stat().st_size / (1024 ** 2),
            "sha256": sha256(path),
            "onnx_check": "PASS",
            "runtime_check": "PASS",
            "max_box_error_px_vs_fp32": max(box_errors),
            "max_score_error_vs_fp32": max(score_errors),
            "mean_score_mae_vs_fp32": float(np.mean(score_maes)),
        })
    (output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "calibration_manifest.txt").write_text(
        "\n".join(path.name for path in calibration_paths) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
