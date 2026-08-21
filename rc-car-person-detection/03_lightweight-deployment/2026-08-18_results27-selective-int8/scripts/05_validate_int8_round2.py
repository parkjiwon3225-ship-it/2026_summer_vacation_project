from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import (
    detections_from_raw,
    find_project_root,
    greedy_match,
    letterbox_image,
    list_images,
    package_root,
    select_evenly,
)


def pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    root = find_project_root()
    pkg = package_root()
    models_dir = pkg / "models"
    results_dir = pkg / "results"
    results_dir.mkdir(exist_ok=True)

    fp32 = models_dir / "results27_640_fp32.onnx"
    candidates = sorted(models_dir.glob("results27_640_int8_round2_*.onnx"))
    if not fp32.is_file():
        raise FileNotFoundError(fp32)
    if not candidates:
        raise FileNotFoundError("Round 2 INT8 models not found. Run 04_quantize_int8_round2.py first.")

    valid_dir = root / "data" / "processed" / "v1_grouped" / "valid" / "images"
    paths = select_evenly(list_images(valid_dir), args.samples)
    tensors = [letterbox_image(p) for p in paths]

    model_paths = [fp32] + candidates
    sessions = {}
    for p in model_paths:
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        sessions[p.stem] = sess

    outputs = {}
    latencies = {}
    zero = np.zeros((1, 3, 480, 640), dtype=np.float32)

    for name, sess in sessions.items():
        inp = sess.get_inputs()[0].name
        for _ in range(args.warmup):
            sess.run(None, {inp: zero})
        model_out = []
        times = []
        for x in tensors:
            t0 = time.perf_counter()
            out = sess.run(None, {inp: x})
            times.append((time.perf_counter() - t0) * 1000.0)
            model_out.append(out)
        outputs[name] = model_out
        latencies[name] = times

    ref_name = fp32.stem
    thresholds = [0.10, 0.15, 0.17, 0.20, 0.25, 0.30]
    rows = []

    for name in sessions:
        raw_score_mae = []
        raw_score_max = []
        raw_box_mae = []
        raw_box_max = []

        if name != ref_name:
            for (rb, rs), (cb, cs) in zip(outputs[ref_name], outputs[name], strict=True):
                raw_score_mae.append(float(np.mean(np.abs(rs - cs))))
                raw_score_max.append(float(np.max(np.abs(rs - cs))))
                raw_box_mae.append(float(np.mean(np.abs(rb - cb))))
                raw_box_max.append(float(np.max(np.abs(rb - cb))))

        for conf in thresholds:
            ref_total = cand_total = matched_total = missed_total = extra_total = 0
            ious = []
            for (rb, rs), (cb, cs) in zip(outputs[ref_name], outputs[name], strict=True):
                ref_det = detections_from_raw(rb[0], rs[0], conf)
                cand_det = detections_from_raw(cb[0], cs[0], conf)
                matches, missed, extra = greedy_match(ref_det, cand_det, 0.5)
                ref_total += len(ref_det)
                cand_total += len(cand_det)
                matched_total += len(matches)
                missed_total += missed
                extra_total += extra
                ious.extend(m[2] for m in matches)

            rows.append({
                "model": name,
                "threshold": conf,
                "size_mb": (models_dir / f"{name}.onnx").stat().st_size / 1024 / 1024,
                "mean_ms": float(np.mean(latencies[name])),
                "p95_ms": pct(latencies[name], 95),
                "ref": ref_total,
                "cand": cand_total,
                "match": matched_total,
                "miss": missed_total,
                "extra": extra_total,
                "retain_pct": 100.0 * matched_total / max(ref_total, 1),
                "mean_iou": float(np.mean(ious)) if ious else float("nan"),
                "score_mae": float(np.mean(raw_score_mae)) if raw_score_mae else 0.0,
                "score_max": float(np.max(raw_score_max)) if raw_score_max else 0.0,
                "box_mae": float(np.mean(raw_box_mae)) if raw_box_mae else 0.0,
                "box_max": float(np.max(raw_box_max)) if raw_box_max else 0.0,
            })

    csv_path = results_dir / "int8_round2_preservation.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("=" * 150)
    print("RESULTS.27 INT8 ROUND 2 — FP32 PRESERVATION")
    print("=" * 150)
    print("samples:", len(paths), "threads:", args.threads)
    print()
    print(f"{'model':48s} {'thr':>5} {'MB':>7} {'ms':>8} {'P95':>8} {'ref':>6} {'cand':>6} {'match':>6} {'miss':>6} {'extra':>6} {'retain%':>8} {'IoU':>7} {'scoreMAE':>10}")
    print("-" * 150)
    for r in rows:
        print(
            f"{r['model'][:48]:48s} {r['threshold']:5.2f} {r['size_mb']:7.3f} "
            f"{r['mean_ms']:8.2f} {r['p95_ms']:8.2f} {r['ref']:6d} {r['cand']:6d} "
            f"{r['match']:6d} {r['miss']:6d} {r['extra']:6d} {r['retain_pct']:8.2f} "
            f"{r['mean_iou']:7.4f} {r['score_mae']:10.6f}"
        )
    print("-" * 150)
    print("saved:", csv_path)
    print()
    print("판정 기준:")
    print("- 우선 보존율/extra/miss를 본다. 속도는 그 다음이다.")
    print("- 후보가 FP32 검출을 크게 훼손하면 Pi에서 빠르더라도 채택하지 않는다.")
    print("- 최종 FP/FN 평가는 실제 학교 운동장 영상/negative 입력으로 한다.")


if __name__ == "__main__":
    main()
