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


def p95(xs):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=80)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    root = find_project_root()
    pkg = package_root()
    models = pkg / "models"
    results = pkg / "results"
    results.mkdir(exist_ok=True)

    fp32 = models / "results27_640_fp32.onnx"
    candidates = sorted(models.glob("results27_640_int8_round3_*.onnx"))
    if not fp32.exists():
        raise FileNotFoundError(fp32)
    if not candidates:
        raise FileNotFoundError("No Round 3 candidates found.")

    valid = root / "data" / "processed" / "v1_grouped" / "valid" / "images"
    paths = select_evenly(list_images(valid), args.samples)
    tensors = [letterbox_image(p) for p in paths]

    sessions = {}
    for p in [fp32] + candidates:
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        sessions[p.stem] = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])

    raw = {}
    latency = {}
    for name, sess in sessions.items():
        inp = sess.get_inputs()[0]
        shape = [int(v) if isinstance(v, int) else 1 for v in inp.shape]
        zero = np.zeros(shape, dtype=np.float32)
        for _ in range(args.warmup):
            sess.run(None, {inp.name: zero})

        outs, times = [], []
        for x in tensors:
            t0 = time.perf_counter()
            y = sess.run(None, {inp.name: x})
            times.append((time.perf_counter() - t0) * 1000)
            outs.append(y)
        raw[name] = outs
        latency[name] = times

    ref = fp32.stem
    thresholds = [0.10, 0.15, 0.17, 0.20, 0.25, 0.30]
    rows = []

    for name in sessions:
        score_mae = score_max = box_mae = box_max = 0.0
        if name != ref:
            smae, smax, bmae, bmax = [], [], [], []
            for (rb, rs), (cb, cs) in zip(raw[ref], raw[name], strict=True):
                smae.append(float(np.mean(np.abs(rs - cs))))
                smax.append(float(np.max(np.abs(rs - cs))))
                bmae.append(float(np.mean(np.abs(rb - cb))))
                bmax.append(float(np.max(np.abs(rb - cb))))
            score_mae = float(np.mean(smae))
            score_max = float(np.max(smax))
            box_mae = float(np.mean(bmae))
            box_max = float(np.max(bmax))

        for thr in thresholds:
            ref_n = cand_n = match_n = miss_n = extra_n = 0
            ious = []
            for (rb, rs), (cb, cs) in zip(raw[ref], raw[name], strict=True):
                rd = detections_from_raw(rb[0], rs[0], thr)
                cd = detections_from_raw(cb[0], cs[0], thr)
                matches, miss, extra = greedy_match(rd, cd, 0.5)
                ref_n += len(rd)
                cand_n += len(cd)
                match_n += len(matches)
                miss_n += miss
                extra_n += extra
                ious.extend(m[2] for m in matches)

            rows.append({
                "model": name,
                "threshold": thr,
                "size_mb": (models / f"{name}.onnx").stat().st_size / (1024 ** 2),
                "mean_ms": float(np.mean(latency[name])),
                "p95_ms": p95(latency[name]),
                "ref": ref_n,
                "cand": cand_n,
                "match": match_n,
                "miss": miss_n,
                "extra": extra_n,
                "retain_pct": 100.0 * match_n / max(ref_n, 1),
                "mean_iou": float(np.mean(ious)) if ious else float("nan"),
                "score_mae": score_mae,
                "score_max": score_max,
                "box_mae": box_mae,
                "box_max": box_max,
            })

    csv_path = results / "int8_round3_preservation.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("=" * 164)
    print("RESULTS.27 INT8 ROUND 3 — SELECTIVE / MIXED PRECISION PRESERVATION")
    print("=" * 164)
    print("samples:", len(paths), "threads:", args.threads)
    print()
    print(f"{'model':50s} {'thr':>5} {'MB':>7} {'ms':>8} {'P95':>8} {'ref':>6} {'cand':>6} {'match':>6} {'miss':>6} {'extra':>6} {'retain%':>8} {'IoU':>7} {'scoreMAE':>10} {'boxMAE':>10}")
    print("-" * 164)
    for r in rows:
        print(
            f"{r['model'][:50]:50s} {r['threshold']:5.2f} {r['size_mb']:7.3f} "
            f"{r['mean_ms']:8.2f} {r['p95_ms']:8.2f} {r['ref']:6d} {r['cand']:6d} "
            f"{r['match']:6d} {r['miss']:6d} {r['extra']:6d} {r['retain_pct']:8.2f} "
            f"{r['mean_iou']:7.4f} {r['score_mae']:10.6f} {r['box_mae']:10.4f}"
        )
    print("-" * 164)
    print("saved:", csv_path)
    print()
    print("판정 가이드:")
    print("  >= 95% retain : 강한 후보")
    print("  90~95%        : 현장 검증 가치 있음")
    print("  < 90%         : 우선 탈락")
    print("단, extra/miss와 실제 운동장 FP/FN을 함께 확인해야 최종 채택할 수 있습니다.")


if __name__ == "__main__":
    main()
