from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common import (
    find_project_root,
    letterbox_image,
    list_images,
    package_root,
    select_evenly,
)


class Reader(CalibrationDataReader):
    def __init__(self, paths: list[Path], input_name: str):
        self.paths = list(paths)
        self.input_name = input_name
        self.rewind()

    def rewind(self):
        self._iter = iter(self.paths)

    def get_next(self):
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        return {self.input_name: letterbox_image(path)}


def check_runtime(path: Path):
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.zeros((1, 3, 480, 640), dtype=np.float32)
    outputs = sess.run(None, {inp.name: x})
    return inp.name, inp.shape, [tuple(v.shape) for v in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-samples", type=int, default=400)
    parser.add_argument("--percentile", type=float, default=99.99)
    args = parser.parse_args()

    root = find_project_root()
    pkg = package_root()
    models_dir = pkg / "models"
    results_dir = pkg / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    fp32 = models_dir / "results27_640_fp32.onnx"
    if not fp32.is_file():
        raise FileNotFoundError(f"FP32 ONNX가 없습니다: {fp32}")

    train_dir = root / "data" / "processed" / "v1_grouped" / "train" / "images"
    images = list_images(train_dir)
    selected = select_evenly(images, min(args.calibration_samples, len(images)))
    if len(selected) < 50:
        raise RuntimeError(f"Calibration image too few: {len(selected)}")

    # ORT recommends preprocessing before quantization so tensor shapes/fusions are known.
    preprocessed = models_dir / "results27_640_fp32_preprocessed.onnx"
    print("=" * 96)
    print("RESULTS.27 INT8 ROUND 2")
    print("=" * 96)
    print("FP32               :", fp32)
    print("calibration source :", train_dir)
    print("samples            :", len(selected))
    print("percentile         :", args.percentile)
    print()

    print("[PREPROCESS] quant_pre_process")
    t0 = time.perf_counter()
    quant_pre_process(
        input_model_path=str(fp32),
        output_model_path=str(preprocessed),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
    )
    print(f"[OK] {preprocessed.name} ({time.perf_counter()-t0:.2f}s)")

    base_session = ort.InferenceSession(str(preprocessed), providers=["CPUExecutionProvider"])
    input_name = base_session.get_inputs()[0].name

    # Round 1 already tried S8S8 QDQ MinMax/Percentile.
    # Round 2 targets materially different data-type/calibration combinations.
    variants = [
        {
            "tag": "s8s8_conv_entropy",
            "activation": QuantType.QInt8,
            "weight": QuantType.QInt8,
            "method": CalibrationMethod.Entropy,
            "extra": {},
        },
        {
            "tag": "u8u8_conv_minmax",
            "activation": QuantType.QUInt8,
            "weight": QuantType.QUInt8,
            "method": CalibrationMethod.MinMax,
            "extra": {},
        },
        {
            "tag": "u8u8_conv_percentile",
            "activation": QuantType.QUInt8,
            "weight": QuantType.QUInt8,
            "method": CalibrationMethod.Percentile,
            "extra": {"CalibPercentile": float(args.percentile)},
        },
        {
            "tag": "u8s8_conv_minmax",
            "activation": QuantType.QUInt8,
            "weight": QuantType.QInt8,
            "method": CalibrationMethod.MinMax,
            "extra": {},
        },
        {
            "tag": "u8s8_conv_percentile",
            "activation": QuantType.QUInt8,
            "weight": QuantType.QInt8,
            "method": CalibrationMethod.Percentile,
            "extra": {"CalibPercentile": float(args.percentile)},
        },
    ]

    report = {
        "fp32": str(fp32),
        "preprocessed": str(preprocessed),
        "calibration_samples": len(selected),
        "percentile": args.percentile,
        "variants": [],
    }

    for v in variants:
        out = models_dir / f"results27_640_int8_round2_{v['tag']}.onnx"

        # Entropy calibration buffers significantly more activation data than
        # MinMax/Percentile and can exhaust RAM at 640x480. Limit only the
        # entropy candidate; keep the full requested sample count for all
        # other variants.
        variant_paths = selected
        if v["method"] == CalibrationMethod.Entropy:
            entropy_count = min(64, len(selected))
            variant_paths = select_evenly(selected, entropy_count)

        reader = Reader(variant_paths, input_name)
        print(f"\n[START] {v['tag']} (calibration={len(variant_paths)})")
        t0 = time.perf_counter()
        try:
            quantize_static(
                model_input=str(preprocessed),
                model_output=str(out),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                reduce_range=False,
                activation_type=v["activation"],
                weight_type=v["weight"],
                calibrate_method=v["method"],
                calibration_providers=["CPUExecutionProvider"],
                op_types_to_quantize=["Conv"],
                extra_options=v["extra"],
            )
            elapsed = time.perf_counter() - t0
            inp_name, inp_shape, out_shapes = check_runtime(out)
            size_mb = out.stat().st_size / 1024 / 1024
            print(f"[OK] {out.name}")
            print(f"     size={size_mb:.3f} MB quant_sec={elapsed:.1f}")
            print(f"     input={inp_name} {inp_shape} outputs={out_shapes}")
            report["variants"].append({
                "tag": v["tag"],
                "file": out.name,
                "size_mb": size_mb,
                "quant_sec": elapsed,
                "activation_type": str(v["activation"]),
                "weight_type": str(v["weight"]),
                "calibration_method": str(v["method"]),
                "calibration_samples_used": len(variant_paths),
                "status": "PASS",
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[FAIL] {v['tag']}: {type(exc).__name__}: {exc}")
            report["variants"].append({
                "tag": v["tag"],
                "activation_type": str(v["activation"]),
                "weight_type": str(v["weight"]),
                "calibration_method": str(v["method"]),
                "calibration_samples_used": len(variant_paths),
                "quant_sec": elapsed,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            })
            if out.exists():
                out.unlink()

    report_path = results_dir / "int8_round2_build_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = results_dir / "int8_round2_calibration_manifest.txt"
    manifest.write_text("\n".join(str(p) for p in selected) + "\n", encoding="utf-8")

    print("\n" + "=" * 96)
    print("[OK] Round 2 quantization complete")
    print("report:", report_path)
    print("next  : python scripts\\05_validate_int8_round2.py --samples 60 --threads 2")
    print("=" * 96)


if __name__ == "__main__":
    main()
