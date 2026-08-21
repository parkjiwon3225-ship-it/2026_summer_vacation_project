from __future__ import annotations

import argparse
import json
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
        self._it = iter(self.paths)

    def get_next(self):
        try:
            p = next(self._it)
        except StopIteration:
            return None
        return {self.input_name: letterbox_image(p)}


def is_backbone(name: str) -> bool:
    n = (name or "").lower()
    return "/backbone/" in n or "backbone/" in n


def split_four(seq):
    n = len(seq)
    cuts = [0, round(n * .25), round(n * .50), round(n * .75), n]
    return [seq[cuts[i]:cuts[i+1]] for i in range(4)]


def runtime_check(path: Path):
    onnx.checker.check_model(onnx.load(path))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [int(v) if isinstance(v, int) else 1 for v in inp.shape]
    outs = sess.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})
    return inp.shape, [tuple(x.shape) for x in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-samples", type=int, default=128)
    ap.add_argument("--percentile", type=float, default=99.99)
    args = ap.parse_args()

    root = find_project_root()
    pkg = package_root()
    models = pkg / "models"
    results = pkg / "results"
    models.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    fp32 = models / "results27_640_fp32.onnx"
    if not fp32.exists():
        raise FileNotFoundError(fp32)

    train = root / "data" / "processed" / "v1_grouped" / "train" / "images"
    images = list_images(train)
    selected = select_evenly(images, min(args.calibration_samples, len(images)))

    pre = models / "results27_640_fp32_preprocessed_round5.onnx"
    print("=" * 112)
    print("RESULTS.27 INT8 ROUND 5 — SAFE BACKBONE MIX")
    print("=" * 112)
    print("FP32        :", fp32)
    print("calibration :", len(selected))
    print("percentile  :", args.percentile)

    quant_pre_process(
        input_model_path=str(fp32),
        output_model_path=str(pre),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
    )

    model = onnx.load(pre)
    backbone = [n.name for n in model.graph.node if n.op_type == "Conv" and is_backbone(n.name)]
    if not backbone:
        raise RuntimeError("No backbone Conv nodes found.")

    q1, q2, q3, q4 = split_four(backbone)

    # Round 4 showed Q1 is highly sensitive, while Q2/Q3/Q4 are individually safe.
    # Test interactions while keeping Q1 completely FP32.
    variants = [
        ("q2_q3", q2 + q3),
        ("q2_q4", q2 + q4),
        ("q2_q3_q4", q2 + q3 + q4),
    ]

    base = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"])
    input_name = base.get_inputs()[0].name

    report = {
        "backbone_conv_count": len(backbone),
        "quarter_sizes": [len(q1), len(q2), len(q3), len(q4)],
        "calibration_samples": len(selected),
        "percentile": args.percentile,
        "q1_policy": "FP32 only (Round 4 sensitivity failure)",
        "variants": [],
    }

    for tag, nodes in variants:
        out_path = models / f"results27_640_int8_round5_{tag}.onnx"
        print("\n" + "-" * 112)
        print(f"[START] {tag} | quantized backbone nodes={len(nodes)}")
        reader = Reader(selected, input_name)
        t0 = time.perf_counter()
        try:
            quantize_static(
                model_input=str(pre),
                model_output=str(out_path),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
                calibrate_method=CalibrationMethod.Percentile,
                calibration_providers=["CPUExecutionProvider"],
                op_types_to_quantize=["Conv"],
                nodes_to_quantize=nodes,
                extra_options={"CalibPercentile": float(args.percentile)},
            )
            sec = time.perf_counter() - t0
            inp_shape, out_shapes = runtime_check(out_path)
            mb = out_path.stat().st_size / (1024 ** 2)
            print(f"[OK] {out_path.name}")
            print(f"size={mb:.3f} MB sec={sec:.1f}")
            print("input :", inp_shape)
            print("output:", out_shapes)
            report["variants"].append({
                "tag": tag,
                "status": "PASS",
                "nodes": len(nodes),
                "size_mb": mb,
                "quant_sec": sec,
            })
        except Exception as e:
            sec = time.perf_counter() - t0
            print(f"[FAIL] {type(e).__name__}: {e}")
            if out_path.exists():
                out_path.unlink()
            report["variants"].append({
                "tag": tag,
                "status": "FAIL",
                "nodes": len(nodes),
                "quant_sec": sec,
                "error": f"{type(e).__name__}: {e}",
            })

    report_path = results / "int8_round5_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 112)
    print("saved:", report_path)
    print("next : python scripts\\11_validate_int8_round5.py --samples 120 --threads 2")
    print("=" * 112)


if __name__ == "__main__":
    main()
