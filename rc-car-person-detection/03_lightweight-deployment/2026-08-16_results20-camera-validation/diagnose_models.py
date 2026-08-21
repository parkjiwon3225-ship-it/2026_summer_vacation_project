import argparse
import csv
import os
import platform
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# Models
# ============================================================

MODEL_FILES = {
    "fp32": "results20_fpn48_fp32.onnx",
    "int8_minmax": "results20_fpn48_int8_qdq_minmax.onnx",
    "int8_percentile": "results20_fpn48_int8_qdq_percentile.onnx",
}

INPUT_W = 320
INPUT_H = 240

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "results", "diagnostics")
DEFAULT_SCREENSHOT_DIR = os.path.join(BASE_DIR, "results", "screenshots")


# ============================================================
# Utilities
# ============================================================

def stats(values):
    arr = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def match_detections(reference, candidate, iou_threshold=0.5):
    pairs = []

    for i, ref in enumerate(reference):
        for j, cand in enumerate(candidate):
            iou = box_iou(ref["box"], cand["box"])

            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True)

    used_ref = set()
    used_cand = set()

    matches = []

    for iou, i, j in pairs:
        if i in used_ref:
            continue

        if j in used_cand:
            continue

        used_ref.add(i)
        used_cand.add(j)

        matches.append((i, j, iou))

    missed = [
        i for i in range(len(reference))
        if i not in used_ref
    ]

    extra = [
        j for j in range(len(candidate))
        if j not in used_cand
    ]

    return matches, missed, extra


# ============================================================
# Detector
# ============================================================

class Detector:
    def __init__(self, model_path, threads=None):
        options = ort.SessionOptions()

        if threads is not None:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

        self.warmup()

    def warmup(self, count=10):
        dummy = np.zeros(
            (1, 3, INPUT_H, INPUT_W),
            dtype=np.float32,
        )

        for _ in range(count):
            self.session.run(
                self.output_names,
                {self.input_name: dummy},
            )

    def preprocess(self, frame):
        src_h, src_w = frame.shape[:2]

        scale = min(
            INPUT_W / src_w,
            INPUT_H / src_h,
        )

        resized_w = max(
            1,
            round(src_w * scale),
        )

        resized_h = max(
            1,
            round(src_h * scale),
        )

        pad_x = (INPUT_W - resized_w) // 2
        pad_y = (INPUT_H - resized_h) // 2

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        resized = cv2.resize(
            rgb,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full(
            (INPUT_H, INPUT_W, 3),
            114,
            dtype=np.uint8,
        )

        canvas[
            pad_y:pad_y + resized_h,
            pad_x:pad_x + resized_w
        ] = resized

        image = canvas.astype(np.float32) / 255.0

        tensor = np.transpose(
            image,
            (2, 0, 1),
        )

        tensor = np.expand_dims(
            tensor,
            axis=0,
        ).astype(np.float32)

        return tensor, scale, pad_x, pad_y

    def inference(self, tensor):
        return self.session.run(
            self.output_names,
            {self.input_name: tensor},
        )

    def postprocess(
        self,
        boxes,
        scores,
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height,
        conf_threshold,
        nms_threshold,
    ):
        keep = np.where(
            scores >= conf_threshold
        )[0]

        if len(keep) == 0:
            return []

        boxes = boxes[keep]
        scores = scores[keep]

        x1 = (boxes[:, 0] - pad_x) / scale
        y1 = (boxes[:, 1] - pad_y) / scale
        x2 = (boxes[:, 2] - pad_x) / scale
        y2 = (boxes[:, 3] - pad_y) / scale

        x1 = np.clip(
            x1,
            0,
            original_width,
        )

        y1 = np.clip(
            y1,
            0,
            original_height,
        )

        x2 = np.clip(
            x2,
            0,
            original_width,
        )

        y2 = np.clip(
            y2,
            0,
            original_height,
        )

        rects = np.stack(
            [
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            ],
            axis=1,
        )

        indices = cv2.dnn.NMSBoxes(
            rects.tolist(),
            scores.tolist(),
            conf_threshold,
            nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.asarray(
            indices
        ).flatten()

        results = []

        for index in indices:
            x, y, w, h = rects[index]

            if w <= 0 or h <= 0:
                continue

            results.append({
                "box": (
                    float(x),
                    float(y),
                    float(x + w),
                    float(y + h),
                ),
                "score": float(scores[index]),
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results


# ============================================================
# Capture
# ============================================================

def capture_frames(args):
    source = (
        args.video
        if args.video is not None
        else args.webcam
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"카메라/영상을 열 수 없습니다. source={source}"
        )

    # 카메라 실제 정보 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print()
    print("=" * 70)
    print("CAMERA PREVIEW")
    print("=" * 70)
    print(f"Source     : {source}")
    print(f"Resolution : {width} x {height}")
    print(f"Camera FPS : {fps:.2f}")
    print()
    print("카메라 화면을 확인하세요.")
    print("[SPACE] 300프레임 촬영 시작")
    print("[Q]     취소")
    print("=" * 70)

    window_name = "Benchmark Camera Preview"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    )

    # --------------------------------------------------------
    # 1. 사용자가 화면을 확인할 때까지 Preview
    # --------------------------------------------------------

    while True:
        ok, frame = cap.read()

        if not ok:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError(
                "카메라에서 프레임을 읽지 못했습니다."
            )

        preview = frame.copy()

        cv2.putText(
            preview,
            "SPACE: START BENCHMARK",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            preview,
            "Q: QUIT",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            window_name,
            preview
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            break

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    # --------------------------------------------------------
    # 2. 3초 Countdown
    # --------------------------------------------------------

    print("\n3초 후 촬영 시작...")

    countdown_start = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        elapsed = time.time() - countdown_start

        remaining = 3 - int(elapsed)

        if elapsed >= 3:
            break

        preview = frame.copy()

        cv2.putText(
            preview,
            f"START IN {remaining}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.imshow(
            window_name,
            preview
        )

        cv2.waitKey(1)

    # --------------------------------------------------------
    # 3. 실제 benchmark frame capture
    # --------------------------------------------------------

    print(
        f"[Capture] {args.frames} frames recording..."
    )

    frames = []

    while len(frames) < args.frames:

        ok, frame = cap.read()

        if not ok:
            continue

        frames.append(
            frame.copy()
        )

        preview = frame.copy()

        progress = (
            len(frames) / args.frames
        )

        cv2.putText(
            preview,
            f"RECORDING: {len(frames)} / {args.frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            preview,
            (20, 55),
            (420, 75),
            (255, 255, 255),
            2,
        )

        cv2.rectangle(
            preview,
            (20, 55),
            (
                20 + int(400 * progress),
                75
            ),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            preview,
            "Move / turn / leave frame during recording",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            window_name,
            preview
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # 확인용 샘플 이미지 저장
    if len(frames) > 0:
        cv2.imwrite(
            os.path.join(args.screenshot_dir, "benchmark_first_frame.jpg"),
            frames[0]
        )

        cv2.imwrite(
            os.path.join(args.screenshot_dir, "benchmark_middle_frame.jpg"),
            frames[len(frames) // 2]
        )

        cv2.imwrite(
            os.path.join(args.screenshot_dir, "benchmark_last_frame.jpg"),
            frames[-1]
        )

    cap.release()
    cv2.destroyAllWindows()

    print(
        f"[Capture] acquired = {len(frames)}"
    )

    print(
        "[Saved] benchmark_first_frame.jpg"
    )

    print(
        "[Saved] benchmark_middle_frame.jpg"
    )

    print(
        "[Saved] benchmark_last_frame.jpg"
    )

    return frames


# ============================================================
# Main benchmark
# ============================================================

def benchmark(args):
    frames = capture_frames(args)

    if len(frames) == 0:
        return

    print()
    print("=" * 75)
    print("ENVIRONMENT")
    print("=" * 75)

    print(
        "Python:",
        sys.version.split()[0],
    )

    print(
        "Platform:",
        platform.platform(),
    )

    print(
        "ONNX Runtime:",
        ort.__version__,
    )

    print(
        "Providers:",
        ort.get_available_providers(),
    )

    print(
        "Threads:",
        args.threads,
    )

    print(
        "Conf:",
        args.conf,
    )

    print(
        "NMS:",
        args.nms,
    )

    outputs = {}

    summary_rows = []

    # ========================================================
    # Run each model
    # ========================================================

    for model_name, filename in MODEL_FILES.items():

        model_path = os.path.join(args.model_dir, filename)

        if not os.path.exists(model_path):
            print(
                f"\n[SKIP] {model_path} not found"
            )
            continue

        print(
            f"\n[{model_name}] loading..."
        )

        detector = Detector(
            model_path,
            threads=args.threads,
        )

        model_results = []

        pre_times = []
        infer_times = []
        post_times = []
        total_times = []

        max_scores = []
        detection_counts = []

        for frame_index, frame in enumerate(frames):

            original_h, original_w = frame.shape[:2]

            total_start = time.perf_counter()

            pre_start = time.perf_counter()

            (
                tensor,
                scale,
                pad_x,
                pad_y,
            ) = detector.preprocess(frame)

            pre_end = time.perf_counter()

            infer_start = time.perf_counter()

            boxes, scores = detector.inference(
                tensor
            )

            infer_end = time.perf_counter()

            boxes = boxes[0]
            scores = scores[0]

            post_start = time.perf_counter()

            detections = detector.postprocess(
                boxes,
                scores,
                scale,
                pad_x,
                pad_y,
                original_w,
                original_h,
                args.conf,
                args.nms,
            )

            post_end = time.perf_counter()

            total_end = time.perf_counter()

            pre_ms = (
                pre_end - pre_start
            ) * 1000

            infer_ms = (
                infer_end - infer_start
            ) * 1000

            post_ms = (
                post_end - post_start
            ) * 1000

            total_ms = (
                total_end - total_start
            ) * 1000

            max_score = float(
                np.max(scores)
            )

            pre_times.append(pre_ms)
            infer_times.append(infer_ms)
            post_times.append(post_ms)
            total_times.append(total_ms)

            max_scores.append(
                max_score
            )

            detection_counts.append(
                len(detections)
            )

            model_results.append({
                "detections": detections,
                "max_score": max_score,
            })

        outputs[model_name] = model_results

        infer_stats = stats(
            infer_times
        )

        total_stats = stats(
            total_times
        )

        row = {
            "model": model_name,

            "file_size_kb":
                os.path.getsize(model_path)
                / 1024.0,

            "pre_mean_ms":
                np.mean(pre_times),

            "infer_mean_ms":
                infer_stats["mean"],

            "infer_p50_ms":
                infer_stats["p50"],

            "infer_p95_ms":
                infer_stats["p95"],

            "infer_p99_ms":
                infer_stats["p99"],

            "post_mean_ms":
                np.mean(post_times),

            "total_mean_ms":
                total_stats["mean"],

            "fps":
                1000.0
                / total_stats["mean"],

            "mean_max_score":
                np.mean(max_scores),

            "mean_detections":
                np.mean(
                    detection_counts
                ),

            "frames_with_detection_pct":
                np.mean(
                    np.asarray(
                        detection_counts
                    ) > 0
                ) * 100,
        }

        summary_rows.append(row)

    # ========================================================
    # Print timing table
    # ========================================================

    print()
    print("=" * 120)
    print("MODEL PERFORMANCE")
    print("=" * 120)

    header = (
        f"{'Model':20}"
        f"{'InferMean':>12}"
        f"{'P50':>10}"
        f"{'P95':>10}"
        f"{'P99':>10}"
        f"{'Total':>10}"
        f"{'FPS':>10}"
        f"{'MaxScore':>12}"
        f"{'AvgDet':>10}"
    )

    print(header)
    print("-" * 120)

    for row in summary_rows:
        print(
            f"{row['model']:20}"
            f"{row['infer_mean_ms']:12.3f}"
            f"{row['infer_p50_ms']:10.3f}"
            f"{row['infer_p95_ms']:10.3f}"
            f"{row['infer_p99_ms']:10.3f}"
            f"{row['total_mean_ms']:10.3f}"
            f"{row['fps']:10.2f}"
            f"{row['mean_max_score']:12.4f}"
            f"{row['mean_detections']:10.3f}"
        )

    # ========================================================
    # FP32 vs INT8
    # ========================================================

    pairwise_rows = []

    if "fp32" in outputs:

        fp32 = outputs["fp32"]

        print()
        print("=" * 105)
        print(
            "FP32 -> INT8 DETECTION PRESERVATION"
        )
        print("=" * 105)

        print(
            f"{'Model':22}"
            f"{'FP32det':>10}"
            f"{'INT8det':>10}"
            f"{'Matched':>10}"
            f"{'Missed':>10}"
            f"{'Extra':>10}"
            f"{'Retain%':>10}"
            f"{'IoU':>10}"
            f"{'ScoreΔ':>11}"
        )

        print("-" * 105)

        for candidate_name in [
            "int8_minmax",
            "int8_percentile",
        ]:

            if candidate_name not in outputs:
                continue

            candidate = outputs[
                candidate_name
            ]

            total_fp32 = 0
            total_candidate = 0

            total_match = 0
            total_missed = 0
            total_extra = 0

            ious = []
            score_deltas = []
            score_ratios = []

            for ref_frame, cand_frame in zip(
                fp32,
                candidate,
            ):
                ref_dets = ref_frame[
                    "detections"
                ]

                cand_dets = cand_frame[
                    "detections"
                ]

                (
                    matches,
                    missed,
                    extra,
                ) = match_detections(
                    ref_dets,
                    cand_dets,
                    args.match_iou,
                )

                total_fp32 += len(
                    ref_dets
                )

                total_candidate += len(
                    cand_dets
                )

                total_match += len(
                    matches
                )

                total_missed += len(
                    missed
                )

                total_extra += len(
                    extra
                )

                for ref_index, cand_index, iou in matches:

                    ref_score = ref_dets[
                        ref_index
                    ]["score"]

                    cand_score = cand_dets[
                        cand_index
                    ]["score"]

                    ious.append(
                        iou
                    )

                    score_deltas.append(
                        cand_score
                        - ref_score
                    )

                    if ref_score > 0:
                        score_ratios.append(
                            cand_score
                            / ref_score
                        )

            retention = (
                100.0
                * total_match
                / total_fp32
                if total_fp32 > 0
                else 0.0
            )

            mean_iou = (
                np.mean(ious)
                if ious
                else 0.0
            )

            mean_delta = (
                np.mean(score_deltas)
                if score_deltas
                else 0.0
            )

            mean_ratio = (
                np.mean(score_ratios)
                if score_ratios
                else 0.0
            )

            row = {
                "model":
                    candidate_name,

                "fp32_detections":
                    total_fp32,

                "candidate_detections":
                    total_candidate,

                "matched":
                    total_match,

                "missed":
                    total_missed,

                "extra":
                    total_extra,

                "retention_pct":
                    retention,

                "mean_iou":
                    mean_iou,

                "mean_score_delta":
                    mean_delta,

                "mean_score_ratio":
                    mean_ratio,
            }

            pairwise_rows.append(
                row
            )

            print(
                f"{candidate_name:22}"
                f"{total_fp32:10}"
                f"{total_candidate:10}"
                f"{total_match:10}"
                f"{total_missed:10}"
                f"{total_extra:10}"
                f"{retention:10.2f}"
                f"{mean_iou:10.4f}"
                f"{mean_delta:11.4f}"
            )

    # ========================================================
    # CSV output
    # ========================================================

    with open(
        os.path.join(args.output_dir, "diagnostic_summary.csv"),
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    if pairwise_rows:
        with open(
            os.path.join(args.output_dir, "diagnostic_pairwise.csv"),
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=pairwise_rows[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                pairwise_rows
            )

    print()
    print("=" * 75)
    print("Saved:")
    print("  diagnostic_summary.csv")
    print("  diagnostic_pairwise.csv")
    print("=" * 75)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--webcam",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--video",
        default=None,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="onnx 파일 폴더 (기본: models/current)",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="진단 CSV 저장 폴더",
    )

    parser.add_argument(
        "--screenshot-dir",
        default=DEFAULT_SCREENSHOT_DIR,
        help="벤치마크 확인 이미지 저장 폴더",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.screenshot_dir, exist_ok=True)

    benchmark(args)