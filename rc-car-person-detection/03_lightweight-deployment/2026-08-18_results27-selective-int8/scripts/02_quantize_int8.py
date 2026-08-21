from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    find_project_root,
    letterbox_image,
    list_images,
    package_root,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-samples", type=int, default=200)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:
        print("[ERROR] ONNX Runtime quantization tools missing:", e)
        print("python -m pip install onnx==1.22.0 onnxruntime==1.28.0")
        sys.exit(2)

    root = find_project_root()
    models_dir = package_root() / "models"
    fp32 = models_dir / "results27_640_fp32.onnx"
    if not fp32.is_file():
        raise FileNotFoundError(f"먼저 FP32 export를 실행하세요: {fp32}")

    train_dir = root / "data" / "processed" / "v1_grouped" / "train" / "images"
    images = list_images(train_dir)
    if len(images) < args.calibration_samples:
        raise RuntimeError(
            f"Calibration images insufficient: {len(images)} < {args.calibration_samples}"
        )

    rng = np.random.default_rng(20260811)
    indices = np.sort(
        rng.choice(len(images), size=args.calibration_samples, replace=False)
    )
    selected = [images[int(i)] for i in indices]

    probe = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    input_name = probe.get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self, paths):
            self.paths = list(paths)
            self.rewind()

        def rewind(self):
            self._iter = iter(self.paths)

        def get_next(self):
            try:
                p = next(self._iter)
            except StopIteration:
                return None
            return {input_name: letterbox_image(p)}

    # We deliberately create a conservative Conv-only QDQ candidate because the
    # previous 320 model showed severe background-score shifts after INT8.
    variants = [
        {
            "name": "results27_640_int8_qdq_conv_minmax.onnx",
            "method": CalibrationMethod.MinMax,
            "ops": ["Conv"],
            "extra": {},
        },
        {
            "name": "results27_640_int8_qdq_conv_percentile.onnx",
            "method": CalibrationMethod.Percentile,
            "ops": ["Conv"],
            "extra": {"CalibPercentile": 99.99},
        },
        {
            "name": "results27_640_int8_qdq_full_minmax.onnx",
            "method": CalibrationMethod.MinMax,
            "ops": None,
            "extra": {},
        },
    ]

    print("=" * 78)
    print("RESULTS.27 STATIC INT8 QDQ")
    print("=" * 78)
    print("FP32 model          :", fp32)
    print("calibration source  :", train_dir)
    print("calibration samples :", len(selected))
    print("input               :", input_name, probe.get_inputs()[0].shape)
    print()

    for v in variants:
        out = models_dir / v["name"]
        reader = Reader(selected)
        print("[START]", v["name"])
        t0 = time.perf_counter()
        kwargs = dict(
            model_input=str(fp32),
            model_output=str(out),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=v["method"],
            calibration_providers=["CPUExecutionProvider"],
            extra_options=v["extra"],
        )
        if v["ops"] is not None:
            kwargs["op_types_to_quantize"] = v["ops"]

        quantize_static(**kwargs)
        elapsed = time.perf_counter() - t0

        # Load immediately so a malformed quantized model fails here.
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 3, 480, 640), dtype=np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: dummy})
        print(
            f"[OK] {out.name} "
            f"size={out.stat().st_size/1024/1024:.3f}MB "
            f"time={elapsed:.1f}s "
            f"boxes={outputs[0].shape} scores={outputs[1].shape}"
        )

    print("\n[OK] Quantization complete.")
    print("주의: INT8은 아직 채택된 모델이 아닙니다. 다음 validate 단계에서 FP32 보존성을 확인하세요.")


if __name__ == "__main__":
    main()
