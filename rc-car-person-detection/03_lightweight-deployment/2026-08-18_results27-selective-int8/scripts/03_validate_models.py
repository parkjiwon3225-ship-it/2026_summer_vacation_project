from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    ExportDetector,
    detections_from_raw,
    find_project_root,
    greedy_match,
    letterbox_image,
    list_images,
    load_source_model,
    package_root,
    select_evenly,
)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    import onnxruntime as ort

    root = find_project_root()
    pkg = package_root()
    models_dir = pkg / "models"
    results_dir = pkg / "results"
    results_dir.mkdir(exist_ok=True)

    model_paths = [
        models_dir / "results27_640_fp32.onnx",
        models_dir / "results27_640_int8_qdq_conv_minmax.onnx",
        models_dir / "results27_640_int8_qdq_conv_percentile.onnx",
        models_dir / "results27_640_int8_qdq_full_minmax.onnx",
    ]
    missing = [p for p in model_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing models: " + ", ".join(map(str, missing)))

    valid_dir = root / "data" / "processed" / "v1_grouped" / "valid" / "images"
    samples = select_evenly(list_images(valid_dir), args.samples)
    tensors = [letterbox_image(p) for p in samples]

    sessions = {}
    for p in model_paths:
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        sessions[p.stem] = sess

    # PT vs FP32 raw equivalence on first 3 samples.
    source_model, ckpt, _ = load_source_model()
    wrapper = ExportDetector(source_model).eval()
    fp_sess = sessions["results27_640_fp32"]
    pt_eq = []
    for x in tensors[:3]:
        with torch.no_grad():
            pt_b, pt_s = wrapper(torch.from_numpy(x))
        fp_b, fp_s = fp_sess.run(None, {fp_sess.get_inputs()[0].name: x})
        pt_eq.append(
            {
                "score_max_abs": float(np.max(np.abs(pt_s.numpy() - fp_s))),
                "score_mean_abs": float(np.mean(np.abs(pt_s.numpy() - fp_s))),
                "box_max_abs": float(np.max(np.abs(pt_b.numpy() - fp_b))),
                "box_mean_abs": float(np.mean(np.abs(pt_b.numpy() - fp_b))),
            }
        )

    # Run every model on exactly the same images.
    outputs = {name: [] for name in sessions}
    latency = {name: [] for name in sessions}
    for name, sess in sessions.items():
        input_name = sess.get_inputs()[0].name
        zero = np.zeros((1, 3, 480, 640), dtype=np.float32)
        for _ in range(args.warmup):
            sess.run(None, {input_name: zero})
        for x in tensors:
            t0 = time.perf_counter()
            out = sess.run(None, {input_name: x})
            latency[name].append((time.perf_counter() - t0) * 1000.0)
            outputs[name].append(out)

    fp_name = "results27_640_fp32"
    thresholds = [0.17, 0.25]
    summary_rows = []

    for name, sess in sessions.items():
        raw_score_mae = []
        raw_score_max = []
        raw_box_mae = []
        raw_box_max = []

        if name != fp_name:
            for (fb, fs), (cb, cs) in zip(outputs[fp_name], outputs[name]):
                raw_score_mae.append(float(np.mean(np.abs(fs - cs))))
                raw_score_max.append(float(np.max(np.abs(fs - cs))))
                raw_box_mae.append(float(np.mean(np.abs(fb - cb))))
                raw_box_max.append(float(np.max(np.abs(fb - cb))))

        for conf in thresholds:
            ref_total = cand_total = matched_total = missed_total = extra_total = 0
            ious = []
            for (fb, fs), (cb, cs) in zip(outputs[fp_name], outputs[name]):
                ref = detections_from_raw(fb[0], fs[0], conf)
                cand = detections_from_raw(cb[0], cs[0], conf)
                matches, missed, extra = greedy_match(ref, cand, 0.5)
                ref_total += len(ref)
                cand_total += len(cand)
                matched_total += len(matches)
                missed_total += missed
                extra_total += extra
                ious.extend(m[2] for m in matches)

            summary_rows.append(
                {
                    "model": name,
                    "threshold": conf,
                    "file_size_mb": round((models_dir / (name + ".onnx")).stat().st_size / 1024 / 1024, 4),
                    "infer_mean_ms": round(float(np.mean(latency[name])), 4),
                    "infer_p50_ms": round(percentile(latency[name], 50), 4),
                    "infer_p95_ms": round(percentile(latency[name], 95), 4),
                    "fp32_ref_detections": ref_total,
                    "candidate_detections": cand_total,
                    "matched": matched_total,
                    "missed_vs_fp32": missed_total,
                    "extra_vs_fp32": extra_total,
                    "retention_pct": round(100.0 * matched_total / max(ref_total, 1), 3),
                    "mean_matched_iou": round(float(np.mean(ious)) if ious else float("nan"), 6),
                    "raw_score_mae": round(float(np.mean(raw_score_mae)) if raw_score_mae else 0.0, 8),
                    "raw_score_max_abs": round(float(np.max(raw_score_max)) if raw_score_max else 0.0, 8),
                    "raw_box_mae": round(float(np.mean(raw_box_mae)) if raw_box_mae else 0.0, 6),
                    "raw_box_max_abs": round(float(np.max(raw_box_max)) if raw_box_max else 0.0, 6),
                }
            )

    csv_path = results_dir / "offline_preservation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "source": {
            "epoch": int(ckpt["epoch"]),
            "best_map50_95": float(ckpt["best_map50_95"]),
            "input": [1, 3, 480, 640],
        },
        "samples": len(samples),
        "threads": args.threads,
        "pytorch_vs_fp32": {
            "score_max_abs": max(r["score_max_abs"] for r in pt_eq),
            "score_mean_abs": float(np.mean([r["score_mean_abs"] for r in pt_eq])),
            "box_max_abs": max(r["box_max_abs"] for r in pt_eq),
            "box_mean_abs": float(np.mean([r["box_mean_abs"] for r in pt_eq])),
        },
        "csv": str(csv_path),
    }
    json_path = results_dir / "validation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 128)
    print("RESULTS.27 640 FP32 / INT8 OFFLINE PRESERVATION")
    print("=" * 128)
    print("source epoch:", ckpt["epoch"], "mAP50-95:", ckpt["best_map50_95"])
    print("samples:", len(samples), "threads:", args.threads)
    print("\nPyTorch -> FP32 ONNX:")
    for k, v in report["pytorch_vs_fp32"].items():
        print(f"  {k}: {v}")

    print("\n" + "-" * 128)
    print(
        f"{'model':43s} {'thr':>5s} {'MB':>7s} {'mean ms':>9s} {'P95':>9s} "
        f"{'ref':>7s} {'cand':>7s} {'match':>7s} {'miss':>7s} {'extra':>7s} {'retain%':>9s} {'IoU':>8s}"
    )
    print("-" * 128)
    for r in summary_rows:
        print(
            f"{r['model'][:43]:43s} {r['threshold']:5.2f} {r['file_size_mb']:7.3f} "
            f"{r['infer_mean_ms']:9.3f} {r['infer_p95_ms']:9.3f} "
            f"{r['fp32_ref_detections']:7d} {r['candidate_detections']:7d} "
            f"{r['matched']:7d} {r['missed_vs_fp32']:7d} {r['extra_vs_fp32']:7d} "
            f"{r['retention_pct']:9.2f} {r['mean_matched_iou']:8.4f}"
        )
    print("-" * 128)
    print("saved:", csv_path)
    print("saved:", json_path)
    print("\n주의: 이 단계는 FP32 대비 보존성 검사입니다.")
    print("실제 false positive/fn 판정은 오후 학교 운동장/negative 입력으로 최종 확인해야 합니다.")


if __name__ == "__main__":
    main()
