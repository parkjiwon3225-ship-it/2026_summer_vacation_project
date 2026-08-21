"""
results20_fpn48 계열 사람 감지 모델 (FP32 / INT8-minmax / INT8-percentile)
성능 비교 도구 — 라즈베리파이4 탑재 전, 노트북 웹캠으로 사전 점검용.

핵심 구조 (onnx 파일 3개를 모두 열어서 확인한 내용)
- 입력  images : [1, 3, 240, 320]  float32, CHW, H=240 W=320
- 출력  boxes  : [1, 6380, 4]      -> 이미 320x240 픽셀 좌표계의 (x1, y1, x2, y2) 절대좌표
          scores : [1, 6380]        -> 이미 시그모이드 적용된 person 확률(0~1), 클래스는 1개(사람)뿐
- 6380 = (80x60 + 40x30 + 20x15 + 10x8) : stride 4/8/16/32, 앵커 1개/셀 FPN 구조와 정확히 일치
  -> box decode가 그래프 내부에 이미 포함되어 있어서 별도 anchor decode 수식 없이
     바로 스케일만 원본 프레임 크기로 맞춰주면 됩니다.

주의: 학습 시 실제 전처리(정규화 방식, RGB/BGR 순서, mean/std 사용 여부 등)는
모델 파일만으로는 100% 확정할 수 없습니다. 아래 기본값은
"BGR->RGB, 0~1 정규화(255로 나눔), mean/std 없음" 을 가정한 것이며,
실제 검출 결과가 이상하면(계속 박스가 하나도 안 잡히거나 위치가 완전히 어긋나면)
--normalize 옵션과 preprocess() 함수의 주석을 참고해서 바꿔보세요.

사용 예시
---------
# 실시간으로 웹캠에서 3개 모델을 숫자키로 바꿔가며 비교
python detect_compare.py --webcam 0

# 특정 모델 하나만 확인
python detect_compare.py --webcam 0 --model fp32

# 동영상 파일로 확인
python detect_compare.py --video sample.mp4

# 세 모델을 자동으로 돌아가며 N프레임씩 측정해서 표/CSV로 비교 (라즈베리파이4 흉내: 코어 수 제한)
python detect_compare.py --webcam 0 --benchmark --frames 200 --threads 4
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort


# ----------------------------------------------------------------------------
# 모델 정의
# ----------------------------------------------------------------------------
MODEL_FILES = {
    "fp32": "results20_fpn48_fp32.onnx",
    "int8_minmax": "results20_fpn48_int8_qdq_minmax.onnx",
    "int8_percentile": "results20_fpn48_int8_qdq_percentile.onnx",
}
MODEL_LABELS = {
    "fp32": "1: FP32 (원본)",
    "int8_minmax": "2: INT8 QDQ (minmax)",
    "int8_percentile": "3: INT8 QDQ (percentile)",
}
KEY_TO_MODEL = {ord("1"): "fp32", ord("2"): "int8_minmax", ord("3"): "int8_percentile"}

INPUT_W, INPUT_H = 320, 240  # 모델 입력 크기 (W, H) — onnx shape [1,3,240,320]에서 확인


class Detector:
    """onnxruntime 세션 + 전/후처리를 감싸는 래퍼."""

    def __init__(self, model_path, threads=None):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        # CPUExecutionProvider만 사용: 라즈베리파이4는 GPU 가속이 없으므로
        # 노트북에서도 CPU로만 돌려야 실제 배포 환경과 비교가 의미 있습니다.
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._warmup()

    def _warmup(self, n=3):
        dummy = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
        for _ in range(n):
            self.session.run(self.output_names, {self.input_name: dummy})

    def preprocess(self, frame_bgr):
        # 단순 리사이즈(letterbox 없음): 학습 때도 종횡비 보존 없이 바로
        # 320x240으로 리사이즈했다고 가정. 웹캠이 16:9라면 약간 눌려 보일 수 있음.
        resized = cv2.resize(frame_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= 255.0  # 0~1 정규화 (mean/std 미적용 가정)
        chw = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def infer(self, frame_bgr):
        inp = self.preprocess(frame_bgr)
        t0 = time.perf_counter()
        boxes, scores = self.session.run(self.output_names, {self.input_name: inp})
        t1 = time.perf_counter()
        return boxes[0], scores[0], (t1 - t0)

    def postprocess(self, boxes, scores, orig_w, orig_h, conf_th=0.5, nms_th=0.4):
        keep_idx = np.where(scores >= conf_th)[0]
        if len(keep_idx) == 0:
            return []

        b = boxes[keep_idx]
        s = scores[keep_idx]

        scale_x = orig_w / INPUT_W
        scale_y = orig_h / INPUT_H
        x1 = b[:, 0] * scale_x
        y1 = b[:, 1] * scale_y
        x2 = b[:, 2] * scale_x
        y2 = b[:, 3] * scale_y

        # OpenCV NMS는 (x, y, w, h) 형식을 받음
        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        rects_list = rects.tolist()
        scores_list = s.tolist()

        indices = cv2.dnn.NMSBoxes(rects_list, scores_list, conf_th, nms_th)
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        results = []
        for i in indices:
            x, y, w, h = rects_list[i]
            results.append(
                {
                    "box": (int(x), int(y), int(x + w), int(y + h)),
                    "score": float(scores_list[i]),
                }
            )
        return results


# ----------------------------------------------------------------------------
# 그리기 / 유틸
# ----------------------------------------------------------------------------
def draw_detections(frame, detections, color=(0, 255, 0)):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"person {det['score']:.2f}"
        cv2.putText(
            frame, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )


def draw_hud(frame, model_key, fps, infer_ms, n_det):
    lines = [
        f"Model: {MODEL_LABELS[model_key]}",
        f"FPS(total): {fps:5.1f}   Inference: {infer_ms:5.1f} ms",
        f"Detections: {n_det}",
        "[1] FP32  [2] INT8-minmax  [3] INT8-percentile   [q] Quit  [s] Save",
    ]
    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
        y += 22


def resolve_model_dir(model_dir):
    paths = {}
    for key, fname in MODEL_FILES.items():
        p = os.path.join(model_dir, fname)
        if not os.path.exists(p):
            print(f"[경고] 모델 파일을 찾을 수 없습니다: {p}")
        paths[key] = p
    return paths


# ----------------------------------------------------------------------------
# 실시간 비교 모드
# ----------------------------------------------------------------------------
def run_live(args, model_paths):
    cap = cv2.VideoCapture(args.video if args.video else args.webcam)
    if not cap.isOpened():
        print("[오류] 카메라/영상을 열 수 없습니다.")
        sys.exit(1)

    current_key = args.model if args.model else "fp32"
    detectors = {}  # 지연 로딩 (선택된 모델만 우선 로드)

    def get_detector(key):
        if key not in detectors:
            print(f"[로딩] {key} 모델 로딩 중...")
            detectors[key] = Detector(model_paths[key], threads=args.threads)
        return detectors[key]

    get_detector(current_key)

    fps_window = deque(maxlen=30)
    stats = {k: {"frames": 0, "total_infer_ms": 0.0} for k in MODEL_FILES}

    print("실시간 비교 시작. 창에서 1/2/3 키로 모델 전환, q로 종료, s로 스크린샷 저장.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 더 이상 읽을 수 없습니다.")
            break

        orig_h, orig_w = frame.shape[:2]
        det = get_detector(current_key)

        t_start = time.perf_counter()
        boxes, scores, infer_time = det.infer(frame)
        results = det.postprocess(boxes, scores, orig_w, orig_h,
                                   conf_th=args.conf, nms_th=args.nms)
        t_end = time.perf_counter()

        fps_window.append(1.0 / max(t_end - t_start, 1e-6))
        fps = sum(fps_window) / len(fps_window)

        stats[current_key]["frames"] += 1
        stats[current_key]["total_infer_ms"] += infer_time * 1000.0

        draw_detections(frame, results)
        draw_hud(frame, current_key, fps, infer_time * 1000.0, len(results))

        cv2.imshow("Person Detector Comparison (RPi4 target)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"capture_{current_key}_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[저장] {fname}")
        elif key in KEY_TO_MODEL:
            new_key = KEY_TO_MODEL[key]
            if new_key != current_key:
                current_key = new_key
                fps_window.clear()

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== 세션 요약 (모델별 평균 추론 시간) ===")
    for k, v in stats.items():
        if v["frames"] > 0:
            avg = v["total_infer_ms"] / v["frames"]
            print(f"{MODEL_LABELS[k]:35s} frames={v['frames']:5d}  avg_infer={avg:6.2f} ms")


# ----------------------------------------------------------------------------
# 벤치마크 모드: 3개 모델을 동일 프레임 수만큼 순서대로 측정
# ----------------------------------------------------------------------------
def run_benchmark(args, model_paths):
    cap = cv2.VideoCapture(args.video if args.video else args.webcam)
    if not cap.isOpened():
        print("[오류] 카메라/영상을 열 수 없습니다.")
        sys.exit(1)

    # 동일한 조건 비교를 위해 프레임을 미리 args.frames장 캡처해서 재사용
    print(f"[벤치마크] 비교용 프레임 {args.frames}장을 캡처합니다...")
    frames = []
    while len(frames) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.copy())
    cap.release()

    if len(frames) == 0:
        print("[오류] 캡처된 프레임이 없습니다.")
        sys.exit(1)

    print(f"[벤치마크] 실제 확보된 프레임 수: {len(frames)}")

    results_table = []
    for key, path in model_paths.items():
        if not os.path.exists(path):
            print(f"[건너뜀] {key}: 파일 없음 ({path})")
            continue

        print(f"\n[벤치마크] {MODEL_LABELS[key]} 측정 중...")
        det = Detector(path, threads=args.threads)

        infer_times = []
        total_times = []
        det_counts = []

        for frame in frames:
            orig_h, orig_w = frame.shape[:2]
            t_start = time.perf_counter()
            boxes, scores, infer_time = det.infer(frame)
            dets = det.postprocess(boxes, scores, orig_w, orig_h,
                                    conf_th=args.conf, nms_th=args.nms)
            t_end = time.perf_counter()

            infer_times.append(infer_time * 1000.0)
            total_times.append((t_end - t_start) * 1000.0)
            det_counts.append(len(dets))

        infer_times = np.array(infer_times)
        total_times = np.array(total_times)
        det_counts = np.array(det_counts)

        row = {
            "model": key,
            "label": MODEL_LABELS[key],
            "n_frames": len(frames),
            "avg_infer_ms": float(infer_times.mean()),
            "p95_infer_ms": float(np.percentile(infer_times, 95)),
            "avg_total_ms": float(total_times.mean()),
            "avg_fps": float(1000.0 / total_times.mean()),
            "avg_detections": float(det_counts.mean()),
            "file_size_kb": os.path.getsize(path) / 1024.0,
        }
        results_table.append(row)

    # 표 출력
    print("\n" + "=" * 100)
    print(f"{'Model':22s} {'AvgInfer(ms)':>13s} {'P95Infer(ms)':>13s} {'AvgFPS':>8s} "
          f"{'AvgDet':>8s} {'FileSize(KB)':>13s}")
    print("-" * 100)
    for r in results_table:
        print(f"{r['label']:22s} {r['avg_infer_ms']:13.2f} {r['p95_infer_ms']:13.2f} "
              f"{r['avg_fps']:8.2f} {r['avg_detections']:8.2f} {r['file_size_kb']:13.1f}")
    print("=" * 100)

    if results_table:
        out_csv = args.output_csv
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
            writer.writeheader()
            writer.writerows(results_table)
        print(f"\n[저장] 결과가 CSV로 저장되었습니다: {out_csv}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="results20_fpn48 사람 감지 모델 성능 비교 도구")
    p.add_argument("--model-dir", default=".", help="onnx 파일들이 있는 폴더 (기본: 현재 폴더)")
    p.add_argument("--webcam", type=int, default=0, help="웹캠 장치 번호 (기본 0)")
    p.add_argument("--video", type=str, default=None, help="웹캠 대신 사용할 동영상 파일 경로")
    p.add_argument("--model", choices=list(MODEL_FILES.keys()), default="fp32",
                    help="실시간 모드에서 처음에 사용할 모델")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (기본 0.5)")
    p.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold (기본 0.4)")
    p.add_argument("--threads", type=int, default=None,
                    help="onnxruntime intra-op 스레드 수 (라즈베리파이4 흉내내려면 4로 설정)")
    p.add_argument("--benchmark", action="store_true",
                    help="3개 모델을 동일 프레임으로 자동 비교하는 벤치마크 모드")
    p.add_argument("--frames", type=int, default=150, help="벤치마크 모드에서 사용할 프레임 수")
    p.add_argument("--output-csv", type=str, default="benchmark_result.csv",
                    help="벤치마크 결과 CSV 저장 경로")
    return p.parse_args()


def main():
    args = parse_args()
    model_paths = resolve_model_dir(args.model_dir)

    if args.benchmark:
        run_benchmark(args, model_paths)
    else:
        run_live(args, model_paths)


if __name__ == "__main__":
    main()
