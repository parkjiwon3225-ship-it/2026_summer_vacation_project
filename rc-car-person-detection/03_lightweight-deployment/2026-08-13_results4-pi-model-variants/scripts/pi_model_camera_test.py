#!/usr/bin/env python3
"""Visual OV5647 + results.4 model variant test for Raspberry Pi 4.

The program writes only numeric CSV/JSON results. No image or video is saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


def summarize(values):
    if not values:
        return {"mean": None, "median": None, "p95": None,
                "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def nms(boxes, threshold):
    remaining = sorted(boxes, key=lambda box: box[4], reverse=True)
    selected = []
    while remaining:
        best = remaining.pop(0)
        selected.append(best)
        remaining = [box for box in remaining if box_iou(best, box) < threshold]
    return selected


def preprocess(frame):
    start = time.perf_counter()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    source_height, source_width = image.shape[:2]
    scale = min(IMAGE_WIDTH / source_width, IMAGE_HEIGHT / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    pad_x = (IMAGE_WIDTH - resized_width) // 2
    pad_y = (IMAGE_HEIGHT - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 114, dtype=np.uint8)
    image[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None]
    tensor = np.ascontiguousarray(image, dtype=np.float32)
    return tensor, (time.perf_counter() - start) * 1000.0, scale, pad_x, pad_y


def decode(boxes_output, scores_output, threshold, nms_threshold):
    start = time.perf_counter()
    boxes = []
    scores = scores_output[0]
    candidates = np.where(scores >= threshold)[0]
    if len(candidates) > 1000:
        candidates = candidates[np.argsort(scores[candidates])[-1000:]]
    for index in candidates:
        x1, y1, x2, y2 = boxes_output[0, index]
        confidence = float(scores[index])
        x1 = float(np.clip(x1, 0, IMAGE_WIDTH))
        y1 = float(np.clip(y1, 0, IMAGE_HEIGHT))
        x2 = float(np.clip(x2, 0, IMAGE_WIDTH))
        y2 = float(np.clip(y2, 0, IMAGE_HEIGHT))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2, confidence])
    return nms(boxes, nms_threshold), (time.perf_counter() - start) * 1000.0


def temperature_c():
    try:
        value = float(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip())
        return value / 1000.0 if value > 200.0 else value
    except (OSError, ValueError):
        return None


def throttled_status():
    import subprocess
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True,
            text=True, timeout=2, check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class Picamera2Source:
    def __init__(self):
        from picamera2 import Picamera2

        self.last_metadata = {}
        self.camera = Picamera2(camera_num=0)
        configuration = self.camera.create_video_configuration(
            # Picamera2/libcamera names are counter-intuitive: RGB888 is
            # captured as BGR byte order, which is the order OpenCV expects.
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 30.0},
            buffer_count=4,
        )
        self.camera.configure(configuration)
        self.camera.start()
        time.sleep(2.0)

    def read(self):
        request = self.camera.capture_request()
        try:
            frame = request.make_array("main")
            self.last_metadata = request.get_metadata()
        finally:
            request.release()
        return frame is not None, frame

    def release(self):
        self.camera.stop()
        self.camera.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-temperature", type=float, default=75.0)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("pi_variant_results"))
    return parser.parse_args()


def make_session(model_path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def metadata_value(metadata, key):
    value = metadata.get(key)
    if isinstance(value, (tuple, list)):
        return json.dumps(list(value))
    return value


def draw(frame, boxes, threshold, inference_ms, total_ms, temp, scenario,
         scale, pad_x, pad_y, model_name):
    display = frame.copy()
    for x1, y1, x2, y2, confidence in boxes:
        p1 = (
            int(np.clip((x1 - pad_x) / scale, 0, display.shape[1] - 1)),
            int(np.clip((y1 - pad_y) / scale, 0, display.shape[0] - 1)),
        )
        p2 = (
            int(np.clip((x2 - pad_x) / scale, 0, display.shape[1] - 1)),
            int(np.clip((y2 - pad_y) / scale, 0, display.shape[0] - 1)),
        )
        cv2.rectangle(display, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            display, f"person {confidence:.3f}",
            (p1[0], max(20, p1[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2,
        )
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0
    temperature_text = "N/A" if temp is None else f"{temp:.1f}C"
    lines = [
        f"{model_name} | people {len(boxes)} | threshold {threshold:.2f} | scene {scenario}",
        f"inference {inference_ms:.1f} ms | loop {fps:.1f} FPS | temp {temperature_text}",
        "Q quit | [ ] threshold | 0 empty 1 near 2 mid 3 far 4 moving 5 multi",
    ]
    for index, text in enumerate(lines):
        cv2.putText(display, text, (10, 24 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imshow("RC Person Detector - model variant test", display)
    return cv2.waitKey(1) & 0xFF


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = (base.parent / args.output / model_path.stem / stamp).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    frame_path = result_dir / "frame_metrics.csv"
    detection_path = result_dir / "detections.csv"

    print("=" * 72)
    print("OV5647 + RESULTS.4 MODEL VARIANT VISUAL TEST")
    print(f"Model            : {model_path.name}")
    print("No camera frame or video is saved. Numeric reports only.")
    print(f"Result directory : {result_dir}")
    print(f"Duration         : {args.duration:.0f} seconds")
    print(f"Thermal stop     : {args.max_temperature:.1f} C")
    print("=" * 72)

    session = make_session(model_path, args.threads)
    input_name = session.get_inputs()[0].name
    camera = Picamera2Source()
    process = psutil.Process()
    process.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)
    dummy = np.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    for _ in range(args.warmup):
        session.run(None, {input_name: dummy})

    frame_fields = [
        "timestamp", "frame", "scenario", "threshold", "detections",
        "capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
        "total_ms", "loop_fps", "sensor_to_result_ms",
        "temperature_c", "system_cpu_percent", "process_cpu_percent",
        "process_rss_mib", "cpu_frequency_mhz", "sensor_timestamp_ns",
        "frame_wall_clock_ns", "frame_duration_us", "exposure_us",
        "analogue_gain", "digital_gain", "lux", "colour_temperature",
    ]
    detection_fields = [
        "timestamp", "frame", "scenario", "detection", "confidence",
        "model_xmin", "model_ymin", "model_xmax", "model_ymax",
        "camera_xmin", "camera_ymin", "camera_xmax", "camera_ymax",
        "center_x", "center_y", "bottom_center_x", "bottom_center_y",
        "normalized_center_x", "normalized_center_y",
        "normalized_bottom_center_x", "normalized_bottom_center_y",
        "box_width_px", "box_height_px", "size_class",
    ]

    start_temp = temperature_c()
    start_throttled = throttled_status()
    status = "PASS"
    error_message = None
    stop_reason = "duration_complete"
    threshold = args.threshold
    scenario = "unlabeled"
    frame_rows = []
    confidences = []
    detection_heights = []
    frames_with_people = 0
    start = time.perf_counter()

    try:
        with frame_path.open("w", newline="", encoding="utf-8") as frame_file, \
             detection_path.open("w", newline="", encoding="utf-8") as detection_file:
            frame_writer = csv.DictWriter(frame_file, fieldnames=frame_fields)
            detection_writer = csv.DictWriter(detection_file, fieldnames=detection_fields)
            frame_writer.writeheader()
            detection_writer.writeheader()

            while time.perf_counter() - start < args.duration:
                loop_start = time.perf_counter()
                capture_start = time.perf_counter()
                ok, frame = camera.read()
                capture_ms = (time.perf_counter() - capture_start) * 1000.0
                if not ok:
                    raise RuntimeError("Picamera2 frame capture failed")

                tensor, preprocess_ms, scale, pad_x, pad_y = preprocess(frame)
                inference_start = time.perf_counter()
                boxes_output, scores_output = session.run(None, {input_name: tensor})
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
                boxes, postprocess_ms = decode(boxes_output, scores_output, threshold, args.nms)
                total_ms = (time.perf_counter() - loop_start) * 1000.0
                now_ns = time.time_ns()
                now_text = datetime.now().isoformat(timespec="milliseconds")
                metadata = getattr(camera, "last_metadata", {})
                wall_clock_ns = metadata.get("FrameWallClock")
                sensor_to_result_ms = (
                    (now_ns - int(wall_clock_ns)) / 1_000_000.0
                    if wall_clock_ns is not None else None
                )
                temp = temperature_c()
                cpu_frequency = psutil.cpu_freq()
                frame_number = len(frame_rows) + 1

                frame_row = {
                    "timestamp": now_text, "frame": frame_number,
                    "scenario": scenario,
                    "threshold": threshold, "detections": len(boxes),
                    "capture_ms": capture_ms, "preprocess_ms": preprocess_ms,
                    "inference_ms": inference_ms, "postprocess_ms": postprocess_ms,
                    "total_ms": total_ms, "loop_fps": 1000.0 / total_ms,
                    "sensor_to_result_ms": sensor_to_result_ms,
                    "temperature_c": temp,
                    "system_cpu_percent": psutil.cpu_percent(interval=None),
                    "process_cpu_percent": process.cpu_percent(interval=None),
                    "process_rss_mib": process.memory_info().rss / (1024 ** 2),
                    "cpu_frequency_mhz": cpu_frequency.current if cpu_frequency else None,
                    "sensor_timestamp_ns": metadata_value(metadata, "SensorTimestamp"),
                    "frame_wall_clock_ns": wall_clock_ns,
                    "frame_duration_us": metadata_value(metadata, "FrameDuration"),
                    "exposure_us": metadata_value(metadata, "ExposureTime"),
                    "analogue_gain": metadata_value(metadata, "AnalogueGain"),
                    "digital_gain": metadata_value(metadata, "DigitalGain"),
                    "lux": metadata_value(metadata, "Lux"),
                    "colour_temperature": metadata_value(metadata, "ColourTemperature"),
                }
                frame_writer.writerow(frame_row)
                frame_rows.append(frame_row)

                if boxes:
                    frames_with_people += 1
                for detection_index, box in enumerate(boxes, start=1):
                    x1, y1, x2, y2, confidence = box
                    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    width, height = x2 - x1, y2 - y1
                    camera_x1 = float(np.clip((x1 - pad_x) / scale, 0, frame.shape[1]))
                    camera_y1 = float(np.clip((y1 - pad_y) / scale, 0, frame.shape[0]))
                    camera_x2 = float(np.clip((x2 - pad_x) / scale, 0, frame.shape[1]))
                    camera_y2 = float(np.clip((y2 - pad_y) / scale, 0, frame.shape[0]))
                    camera_center_x = (camera_x1 + camera_x2) / 2.0
                    camera_center_y = (camera_y1 + camera_y2) / 2.0
                    size_class = "tiny_lt16" if height < 16 else (
                        "small_16_32" if height < 32 else "regular_ge32"
                    )
                    confidences.append(confidence)
                    detection_heights.append(height)
                    detection_writer.writerow({
                        "timestamp": now_text, "frame": frame_number,
                        "scenario": scenario,
                        "detection": detection_index, "confidence": confidence,
                        "model_xmin": x1, "model_ymin": y1,
                        "model_xmax": x2, "model_ymax": y2,
                        "camera_xmin": camera_x1, "camera_ymin": camera_y1,
                        "camera_xmax": camera_x2, "camera_ymax": camera_y2,
                        "center_x": camera_center_x, "center_y": camera_center_y,
                        "bottom_center_x": camera_center_x,
                        "bottom_center_y": camera_y2,
                        "normalized_center_x": camera_center_x / frame.shape[1],
                        "normalized_center_y": camera_center_y / frame.shape[0],
                        "normalized_bottom_center_x": camera_center_x / frame.shape[1],
                        "normalized_bottom_center_y": camera_y2 / frame.shape[0],
                        "box_width_px": width, "box_height_px": height,
                        "size_class": size_class,
                    })

                if frame_number % 30 == 0:
                    frame_file.flush()
                    detection_file.flush()

                if temp is not None and temp >= args.max_temperature:
                    status = "THERMAL_STOP"
                    stop_reason = f"temperature_reached_{temp:.1f}C"
                    break

                if not args.no_preview:
                    key = draw(
                        frame, boxes, threshold, inference_ms, total_ms, temp,
                        scenario, scale, pad_x, pad_y, model_path.stem,
                    )
                    if key == ord("q"):
                        stop_reason = "user_pressed_q"
                        break
                    if key == ord("["):
                        threshold = max(0.05, threshold - 0.05)
                    elif key == ord("]"):
                        threshold = min(0.95, threshold + 0.05)
                    elif key == ord("0"):
                        scenario = "empty"
                    elif key == ord("1"):
                        scenario = "near"
                    elif key == ord("2"):
                        scenario = "middle"
                    elif key == ord("3"):
                        scenario = "far"
                    elif key == ord("4"):
                        scenario = "moving"
                    elif key == ord("5"):
                        scenario = "multiple_people"
    except Exception as error:
        status = "ERROR"
        stop_reason = "exception"
        error_message = repr(error)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    total_detections = len(confidences)
    size_counts = {
        "tiny_lt16": sum(h < 16 for h in detection_heights),
        "small_16_32": sum(16 <= h < 32 for h in detection_heights),
        "regular_ge32": sum(h >= 32 for h in detection_heights),
    }
    scenario_summary = {}
    for scene in sorted({str(row["scenario"]) for row in frame_rows}):
        scene_rows = [row for row in frame_rows if row["scenario"] == scene]
        scenario_summary[scene] = {
            "frames": len(scene_rows),
            "frames_with_detections": sum(int(row["detections"]) > 0 for row in scene_rows),
            "detections": sum(int(row["detections"]) for row in scene_rows),
            "mean_inference_ms": float(np.mean([
                float(row["inference_ms"]) for row in scene_rows
            ])) if scene_rows else None,
        }
    report = {
        "status": status, "stop_reason": stop_reason, "error": error_message,
        "notice": "Visual camera test is not a ground-truth accuracy evaluation.",
        "model": {"path": str(model_path), "name": model_path.name},
        "started": stamp, "elapsed_seconds": elapsed,
        "frames": len(frame_rows), "frames_with_people": frames_with_people,
        "frames_with_people_percent": (
            100.0 * frames_with_people / len(frame_rows) if frame_rows else None
        ),
        "total_detections": total_detections,
        "detections_per_frame": total_detections / len(frame_rows) if frame_rows else None,
        "confidence": summarize(confidences),
        "box_height_model_pixels": summarize(detection_heights),
        "size_counts": size_counts,
        "scenario_summary": scenario_summary,
        "inference_ms": summarize([float(r["inference_ms"]) for r in frame_rows]),
        "total_ms": summarize([float(r["total_ms"]) for r in frame_rows]),
        "sensor_to_result_ms": summarize([
            float(r["sensor_to_result_ms"]) for r in frame_rows
            if r["sensor_to_result_ms"] is not None
        ]),
        "temperature_c": summarize([
            float(r["temperature_c"]) for r in frame_rows
            if r["temperature_c"] is not None
        ]),
        "start_temperature_c": start_temp,
        "end_temperature_c": temperature_c(),
        "throttled_start": start_throttled,
        "throttled_end": throttled_status(),
        "configuration": vars(args) | {"output": str(args.output)},
    }
    (result_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if error_message:
        (result_dir / "ERROR.txt").write_text(error_message, encoding="utf-8")

    print("\n" + "=" * 72)
    print("MODEL VARIANT CAMERA TEST RESULT")
    print(f"Status                 : {status}")
    print(f"Stop reason            : {stop_reason}")
    print(f"Frames                 : {len(frame_rows)}")
    print(f"Frames with people     : {frames_with_people}")
    print(f"Total detections       : {total_detections}")
    if frame_rows:
        print(f"Mean / P95 inference   : {report['inference_ms']['mean']:.2f} / "
              f"{report['inference_ms']['p95']:.2f} ms")
        print(f"Mean sensor-to-result  : {report['sensor_to_result_ms']['mean']} ms")
        print(f"Peak temperature       : {report['temperature_c']['max']} C")
    print(f"Throttle start / end   : {start_throttled} / {report['throttled_end']}")
    print(f"Send back folder       : {result_dir}")
    print("=" * 72)

    if error_message:
        raise RuntimeError(error_message)


if __name__ == "__main__":
    main()
