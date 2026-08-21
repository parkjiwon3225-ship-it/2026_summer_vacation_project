from __future__ import annotations

import argparse
import json
import time
from collections import Counter
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
            p = next(self._iter)
        except StopIteration:
            return None
        return {self.input_name: letterbox_image(p)}


def group(name: str) -> str:
    n = (name or "").lower()
    if "/head/" in n or "head/" in n:
        return "head"
    if "/fpn/" in n or "fpn/" in n:
        return "fpn"
    if "/backbone/" in n or "backbone/" in n:
        return "backbone"
    if "stem" in n:
        return "stem"
    return "other"


def io_shape(model_path: Path):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    return inp.name, inp.shape, [(o.name, o.shape) for o in sess.get_outputs()]


def check_model(path: Path):
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [int(x) if isinstance(x, int) else 1 for x in inp.shape]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected input shape: {inp.shape}")
    x = np.zeros(shape, dtype=np.float32)
    outs = sess.run(None, {inp.name: x})
    return inp.shape, [tuple(x.shape) for x in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-samples", type=int, default=160)
    ap.add_argument("--percentile", type=float, default=99.99)
    args = ap.parse_args()

    root = find_project_root()
    pkg = package_root()
    models = pkg / "models"
    results = pkg / "results"
    models.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    fp32 = models / "results27_640_fp32.onnx"
    if not fp32.is_file():
        raise FileNotFoundError(fp32)

    train = root / "data" / "processed" / "v1_grouped" / "train" / "images"
    images = list_images(train)
    selected = select_evenly(images, min(args.calibration_samples, len(images)))
    if len(selected) < 50:
        raise RuntimeError(f"Too few calibration images: {len(selected)}")

    pre = models / "results27_640_fp32_preprocessed_round3.onnx"
    print("=" * 110)
    print("RESULTS.27 INT8 ROUND 3 — SELECTIVE / MIXED PRECISION")
    print("=" * 110)
    print("source      :", fp32)
    print("calibration :", train)
    print("samples     :", len(selected))
    print("percentile  :", args.percentile)

    quant_pre_process(
        input_model_path=str(fp32),
        output_model_path=str(pre),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
    )

    model = onnx.load(pre)
    conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    counts = Counter(group(node.name) for node in conv_nodes)
    print("conv groups :", dict(counts))

    backbone = [n.name for n in conv_nodes if group(n.name) == "backbone"]
    fpn = [n.name for n in conv_nodes if group(n.name) == "fpn"]
    head = [n.name for n in conv_nodes if group(n.name) == "head"]
    stem = [n.name for n in conv_nodes if group(n.name) == "stem"]
    other = [n.name for n in conv_nodes if group(n.name) == "other"]

    # Three materially different selective scopes:
    # 1) Backbone only: safest accuracy-first mixed precision.
    # 2) Backbone + FPN: more compute coverage, keep detector head FP32.
    # 3) No-head: quantize every Conv except head, catching stem/other convs too.
    variants = [
        ("backbone_only", backbone),
        ("backbone_fpn", backbone + fpn),
        ("no_head", [n.name for n in conv_nodes if group(n.name) != "head"]),
    ]

    base_sess = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"])
    input_name = base_sess.get_inputs()[0].name

    report = {
        "source": str(fp32),
        "preprocessed": str(pre),
        "calibration_samples": len(selected),
        "percentile": args.percentile,
        "conv_group_counts": dict(counts),
        "variants": [],
    }

    for tag, nodes in variants:
        print("\n" + "-" * 110)
        print(f"[START] {tag}")
        print(f"nodes_to_quantize={len(nodes)}")
        if not nodes:
            print("[SKIP] no matching nodes")
            report["variants"].append({"tag": tag, "status": "SKIP", "reason": "no matching nodes"})
            continue

        out_path = models / f"results27_640_int8_round3_{tag}.onnx"
        reader = Reader(selected, input_name)
        t0 = time.perf_counter()
        try:
            quantize_static(
                model_input=str(pre),
                model_output=str(out_path),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                reduce_range=False,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.Percentile,
                calibration_providers=["CPUExecutionProvider"],
                op_types_to_quantize=["Conv"],
                nodes_to_quantize=nodes,
                extra_options={"CalibPercentile": float(args.percentile)},
            )
            sec = time.perf_counter() - t0
            inp_shape, out_shapes = check_model(out_path)
            size_mb = out_path.stat().st_size / (1024 ** 2)
            print(f"[OK] {out_path.name}")
            print(f"size={size_mb:.3f} MB  sec={sec:.1f}")
            print("input :", inp_shape)
            print("output:", out_shapes)
            report["variants"].append({
                "tag": tag,
                "status": "PASS",
                "file": out_path.name,
                "nodes_quantized": len(nodes),
                "size_mb": size_mb,
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
                "nodes_quantized": len(nodes),
                "quant_sec": sec,
                "error": f"{type(e).__name__}: {e}",
            })

    path = results / "int8_round3_build_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 110)
    print("saved:", path)
    print("next : python scripts\\07_validate_int8_round3.py --samples 80 --threads 2")
    print("=" * 110)


if __name__ == "__main__":
    main()
