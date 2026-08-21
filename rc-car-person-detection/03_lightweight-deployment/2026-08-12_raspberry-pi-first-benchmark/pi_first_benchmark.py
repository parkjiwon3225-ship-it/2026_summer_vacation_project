#!/usr/bin/env python3
"""First Raspberry Pi benchmark for the temporary FP32/INT8 person detector.

No video or camera frames are written. Only numeric CSV/JSON reports are saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import onnxruntime as ort
import psutil


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
CONFIDENCE_THRESHOLD = 0.20
NMS_THRESHOLD = 0.20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0", help="Camera index or video path")
    parser.add_argument("--camera-backend", choices=("auto", "picamera2", "opencv"),
                        default="auto", help="CSI camera uses picamera2")
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=180.0,
                        help="Seconds per standalone model phase")
    parser.add_argument("--compare-frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--nms", type=float, default=NMS_THRESHOLD)
    parser.add_argument("--preview", action="store_true",
                        help="Show camera preview; no frames are saved")
    parser.add_argument("--output", type=Path, default=Path("pi_test_results"))
    return parser.parse_args()


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values):
    if not values:
        return {"mean": None, "median": None, "p95": None,
                "min": None, "max": None}
    return {
        "mean": float(mean(values)),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def nms(boxes, threshold):
    remaining = sorted(boxes, key=lambda b: b[4], reverse=True)
    selected = []
    while remaining:
        best = remaining.pop(0)
        selected.append(best)
        remaining = [b for b in remaining if box_iou(best, b) < threshold]
    return selected


def preprocess(frame):
    start = time.perf_counter()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None]
    tensor = np.ascontiguousarray(image, dtype=np.float32)
    return tensor, (time.perf_counter() - start) * 1000.0


def decode(prediction, threshold, nms_threshold):
    start = time.perf_counter()
    grid_h, grid_w = prediction.shape[1:]
    cell_w, cell_h = IMAGE_WIDTH / grid_w, IMAGE_HEIGHT / grid_h
    objectness = sigmoid(prediction[0])
    ys, xs = np.where(objectness >= threshold)
    boxes = []
    for y, x in zip(ys, xs):
        confidence = float(objectness[y, x])
        cx = (float(x) + float(sigmoid(prediction[1, y, x]))) * cell_w
        cy = (float(y) + float(sigmoid(prediction[2, y, x]))) * cell_h
        width = float(sigmoid(prediction[3, y, x])) * IMAGE_WIDTH
        height = float(sigmoid(prediction[4, y, x])) * IMAGE_HEIGHT
        x1 = float(np.clip(cx - width / 2.0, 0, IMAGE_WIDTH))
        y1 = float(np.clip(cy - height / 2.0, 0, IMAGE_HEIGHT))
        x2 = float(np.clip(cx + width / 2.0, 0, IMAGE_WIDTH))
        y2 = float(np.clip(cy + height / 2.0, 0, IMAGE_HEIGHT))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2, confidence])
    boxes = nms(boxes, nms_threshold)
    return boxes, (time.perf_counter() - start) * 1000.0


def temperature_c():
    paths = [Path("/sys/class/thermal/thermal_zone0/temp")]
    for path in paths:
        try:
            value = float(path.read_text(encoding="utf-8").strip())
            return value / 1000.0 if value > 200.0 else value
        except (OSError, ValueError):
            pass
    return None


def throttled_status():
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True,
            text=True, timeout=2, check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "model")) and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def make_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_ms = (time.perf_counter() - start) * 1000.0
    return session, load_ms


class Picamera2Source:
    def __init__(self, camera_number, width, height, fps):
        from picamera2 import Picamera2

        self.width = width
        self.height = height
        self.fps = fps
        self.last_metadata = {}
        self.camera = Picamera2(camera_num=camera_number)
        configuration = self.camera.create_video_configuration(
            # RGB888 produces BGR byte order for OpenCV in Picamera2.
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": float(fps)},
            buffer_count=4,
        )
        self.camera.configure(configuration)
        self.camera.start()
        time.sleep(2.0)

    def read(self):
        try:
            request = self.camera.capture_request()
            try:
                frame = request.make_array("main")
                self.last_metadata = request.get_metadata()
            finally:
                request.release()
            return frame is not None, frame
        except Exception as error:
            print(f"Picamera2 capture error: {error}")
            return False, None

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if property_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def getBackendName(self):
        return "Picamera2/libcamera"

    def release(self):
        self.camera.stop()
        self.camera.close()


def open_camera(source, width, height, fps, backend):
    if backend in ("auto", "picamera2") and source.isdigit():
        try:
            camera = Picamera2Source(int(source), width, height, fps)
            print("Camera backend: Picamera2/libcamera")
            return camera
        except Exception as error:
            if backend == "picamera2":
                raise RuntimeError(f"Picamera2 camera could not be opened: {error}") from error
            print(f"Picamera2 unavailable; trying OpenCV: {error}")

    parsed_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(parsed_source)
    if not cap.isOpened():
        raise RuntimeError(f"Camera/video could not be opened: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def draw_preview(frame, boxes, label):
    shown = frame.copy()
    sx, sy = shown.shape[1] / IMAGE_WIDTH, shown.shape[0] / IMAGE_HEIGHT
    for x1, y1, x2, y2, confidence in boxes:
        cv2.rectangle(shown, (int(x1 * sx), int(y1 * sy)),
                      (int(x2 * sx), int(y2 * sy)), (0, 255, 0), 2)
        cv2.putText(shown, f"{confidence:.2f}",
                    (int(x1 * sx), max(18, int(y1 * sy) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(shown, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.imshow("Pi first benchmark - Q to stop", shown)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def warmup(cap, sessions, count):
    completed = 0
    while completed < count:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed during warmup")
        tensor, _ = preprocess(frame)
        for session, input_name in sessions:
            session.run(None, {input_name: tensor})
        completed += 1


def standalone_phase(name, cap, session, input_name, duration,
                     threshold, nms_threshold, preview, writer, process):
    print(f"\n[{name}] standalone measurement: {duration:.0f} seconds")
    start_phase = time.perf_counter()
    rows = []
    while time.perf_counter() - start_phase < duration:
        loop_start = time.perf_counter()
        capture_start = time.perf_counter()
        ok, frame = cap.read()
        capture_ms = (time.perf_counter() - capture_start) * 1000.0
        if not ok:
            raise RuntimeError("Camera read failed")
        tensor, preprocess_ms = preprocess(frame)
        infer_start = time.perf_counter()
        output = session.run(None, {input_name: tensor})[0]
        inference_ms = (time.perf_counter() - infer_start) * 1000.0
        boxes, postprocess_ms = decode(output[0], threshold, nms_threshold)
        total_ms = (time.perf_counter() - loop_start) * 1000.0
        row = {
            "phase": name, "frame": len(rows) + 1,
            "capture_ms": capture_ms, "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms, "postprocess_ms": postprocess_ms,
            "total_ms": total_ms, "detections": len(boxes),
            "temperature_c": temperature_c(),
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "process_rss_mib": process.memory_info().rss / (1024 ** 2),
        }
        writer.writerow(row)
        rows.append(row)
        if preview and draw_preview(frame, boxes, name):
            break
    return rows


def greedy_match(fp32_boxes, int8_boxes):
    available = set(range(len(int8_boxes)))
    matches = []
    for fp_box in sorted(fp32_boxes, key=lambda b: b[4], reverse=True):
        if not available:
            break
        best_index = max(available, key=lambda i: box_iou(fp_box, int8_boxes[i]))
        score = box_iou(fp_box, int8_boxes[best_index])
        if score >= 0.10:
            matches.append((fp_box, int8_boxes[best_index], score))
            available.remove(best_index)
    return matches


def comparison_phase(cap, fp32, fp32_name, int8, int8_name, frame_count,
                     threshold, nms_threshold, preview, writer):
    print(f"\n[COMPARE] same-frame comparison: {frame_count} frames")
    rows = []
    for index in range(1, frame_count + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        tensor, _ = preprocess(frame)
        start = time.perf_counter()
        fp_out = fp32.run(None, {fp32_name: tensor})[0]
        fp_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        int_out = int8.run(None, {int8_name: tensor})[0]
        int_ms = (time.perf_counter() - start) * 1000.0
        fp_boxes, _ = decode(fp_out[0], threshold, nms_threshold)
        int_boxes, _ = decode(int_out[0], threshold, nms_threshold)
        matches = greedy_match(fp_boxes, int_boxes)
        ious = [m[2] for m in matches]
        confidence_diffs = [abs(m[0][4] - m[1][4]) for m in matches]
        row = {
            "frame": index, "fp32_inference_ms": fp_ms,
            "int8_inference_ms": int_ms, "fp32_detections": len(fp_boxes),
            "int8_detections": len(int_boxes), "count_agreement": int(len(fp_boxes) == len(int_boxes)),
            "matches": len(matches), "mean_iou": mean(ious) if ious else "",
            "mean_confidence_difference": mean(confidence_diffs) if confidence_diffs else "",
        }
        writer.writerow(row)
        rows.append(row)
        if preview and draw_preview(frame, fp_boxes, "COMPARE (FP32 boxes)"):
            break
    return rows


def phase_summary(rows):
    result = {"frames": len(rows)}
    for key in ("capture_ms", "preprocess_ms", "inference_ms",
                "postprocess_ms", "total_ms", "temperature_c",
                "system_cpu_percent", "process_rss_mib"):
        values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        result[key] = summarize(values)
    result["end_to_end_fps_from_mean_total"] = (
        1000.0 / result["total_ms"]["mean"] if result["total_ms"]["mean"] else None
    )
    result["model_fps_from_mean_inference"] = (
        1000.0 / result["inference_ms"]["mean"] if result["inference_ms"]["mean"] else None
    )
    result["detections_total"] = sum(int(r["detections"]) for r in rows)
    return result


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent
    fp32_path = base / "person_detector_fp32.onnx"
    int8_path = base / "person_detector_int8.onnx"
    for path in (fp32_path, int8_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (base / args.output / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("RASPBERRY PI FIRST FP32 / INT8 TEST")
    print("No video or image is saved; numeric reports only.")
    print(f"Output: {output_dir}")
    print("=" * 72)

    fp32, fp32_load_ms = make_session(fp32_path, args.threads)
    int8, int8_load_ms = make_session(int8_path, args.threads)
    fp32_input = fp32.get_inputs()[0].name
    int8_input = int8.get_inputs()[0].name
    cap = open_camera(args.camera, args.capture_width, args.capture_height,
                      args.capture_fps, args.camera_backend)
    actual_camera = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps_reported": float(cap.get(cv2.CAP_PROP_FPS)),
        "backend": cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown",
    }
    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)
    start_temperature = temperature_c()
    start_throttled = throttled_status()

    frame_fields = ["phase", "frame", "capture_ms", "preprocess_ms",
                    "inference_ms", "postprocess_ms", "total_ms", "detections",
                    "temperature_c", "system_cpu_percent", "process_rss_mib"]
    compare_fields = ["frame", "fp32_inference_ms", "int8_inference_ms",
                      "fp32_detections", "int8_detections", "count_agreement",
                      "matches", "mean_iou", "mean_confidence_difference"]
    try:
        warmup(cap, [(fp32, fp32_input), (int8, int8_input)], args.warmup)
        with (output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            frame_writer = csv.DictWriter(handle, fieldnames=frame_fields)
            frame_writer.writeheader()
            fp_rows = standalone_phase("FP32", cap, fp32, fp32_input, args.duration,
                                       args.threshold, args.nms, args.preview,
                                       frame_writer, process)
            int_rows = standalone_phase("INT8", cap, int8, int8_input, args.duration,
                                        args.threshold, args.nms, args.preview,
                                        frame_writer, process)
        with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            compare_writer = csv.DictWriter(handle, fieldnames=compare_fields)
            compare_writer.writeheader()
            comparison = comparison_phase(cap, fp32, fp32_input, int8, int8_input,
                                          args.compare_frames, args.threshold, args.nms,
                                          args.preview, compare_writer)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    matched_ious = [float(r["mean_iou"]) for r in comparison if r["mean_iou"] != ""]
    conf_diffs = [float(r["mean_confidence_difference"]) for r in comparison
                  if r["mean_confidence_difference"] != ""]
    report = {
        "notice": "Temporary pre-final model; use this run for Pi deployment-path testing only.",
        "timestamp": timestamp,
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu_model(), "python": platform.python_version(),
            "opencv": cv2.__version__, "onnxruntime": ort.__version__,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "ram_total_mib": psutil.virtual_memory().total / (1024 ** 2),
            "threads": args.threads, "providers": fp32.get_providers(),
            "camera": actual_camera,
        },
        "configuration": vars(args) | {"output": str(args.output)},
        "models": {
            "FP32": {"bytes": fp32_path.stat().st_size, "load_ms": fp32_load_ms,
                     "input": str(fp32.get_inputs()[0].shape),
                     "output": str(fp32.get_outputs()[0].shape)},
            "INT8": {"bytes": int8_path.stat().st_size, "load_ms": int8_load_ms,
                     "input": str(int8.get_inputs()[0].shape),
                     "output": str(int8.get_outputs()[0].shape)},
        },
        "temperature_start_c": start_temperature,
        "temperature_end_c": temperature_c(),
        "throttled_start": start_throttled,
        "throttled_end": throttled_status(),
        "standalone": {"FP32": phase_summary(fp_rows), "INT8": phase_summary(int_rows)},
        "comparison": {
            "frames": len(comparison),
            "detection_count_agreement_percent": 100.0 * mean(
                [int(r["count_agreement"]) for r in comparison]
            ) if comparison else None,
            "fp32_detections_total": sum(int(r["fp32_detections"]) for r in comparison),
            "int8_detections_total": sum(int(r["int8_detections"]) for r in comparison),
            "matched_box_mean_iou": mean(matched_ious) if matched_ious else None,
            "matched_box_mean_confidence_difference": mean(conf_diffs) if conf_diffs else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fp = report["standalone"]["FP32"]
    iq = report["standalone"]["INT8"]
    print("\n" + "=" * 72)
    print("RESULT")
    for name, item in (("FP32", fp), ("INT8", iq)):
        print(f"{name:5s} inference mean/P95 : {item['inference_ms']['mean']:.2f} / "
              f"{item['inference_ms']['p95']:.2f} ms")
        print(f"{name:5s} end-to-end FPS     : {item['end_to_end_fps_from_mean_total']:.2f}")
        print(f"{name:5s} peak temp / RSS    : {item['temperature_c']['max']} C / "
              f"{item['process_rss_mib']['max']:.1f} MiB")
    print(f"Count agreement             : {report['comparison']['detection_count_agreement_percent']:.2f}%")
    print(f"Mean matched bbox IoU       : {report['comparison']['matched_box_mean_iou']}")
    print(f"Throttle start / end        : {report['throttled_start']} / "
          f"{report['throttled_end']}")
    print(f"Send back this folder       : {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
