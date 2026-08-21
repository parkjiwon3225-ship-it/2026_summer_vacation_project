import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# Configuration
# ============================================================

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

CONF_THRESHOLD = 0.20
NMS_THRESHOLD = 0.20

WARMUP_FRAMES = 20

BASE_DIR = Path(__file__).resolve().parent

FP32_MODEL = BASE_DIR / "person_detector_fp32.onnx"
INT8_MODEL = BASE_DIR / "person_detector_int8.onnx"


# ============================================================
# Utility
# ============================================================

def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area1 = (
        max(0.0, box1[2] - box1[0])
        * max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0])
        * max(0.0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def nms(boxes, threshold):
    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda box: box[4],
        reverse=True,
    )

    selected = []

    while boxes:
        best = boxes.pop(0)
        selected.append(best)

        boxes = [
            box for box in boxes
            if calculate_iou(best[:4], box[:4]) < threshold
        ]

    return selected


# ============================================================
# Preprocess
# ============================================================

def preprocess(frame):
    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
    )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    image = np.expand_dims(
        image,
        axis=0,
    )

    return np.ascontiguousarray(
        image,
        dtype=np.float32,
    )


# ============================================================
# Decode
# ============================================================

def decode_prediction(prediction, threshold):
    grid_h = prediction.shape[1]
    grid_w = prediction.shape[2]

    cell_w = IMAGE_WIDTH / grid_w
    cell_h = IMAGE_HEIGHT / grid_h

    objectness = sigmoid(
        prediction[0]
    )

    ys, xs = np.where(
        objectness >= threshold
    )

    boxes = []

    for y, x in zip(ys, xs):
        confidence = float(
            objectness[y, x]
        )

        tx = float(
            sigmoid(
                prediction[1, y, x]
            )
        )

        ty = float(
            sigmoid(
                prediction[2, y, x]
            )
        )

        bw = float(
            sigmoid(
                prediction[3, y, x]
            )
        ) * IMAGE_WIDTH

        bh = float(
            sigmoid(
                prediction[4, y, x]
            )
        ) * IMAGE_HEIGHT

        cx = (
            float(x) + tx
        ) * cell_w

        cy = (
            float(y) + ty
        ) * cell_h

        xmin = np.clip(
            cx - bw / 2,
            0,
            IMAGE_WIDTH,
        )

        ymin = np.clip(
            cy - bh / 2,
            0,
            IMAGE_HEIGHT,
        )

        xmax = np.clip(
            cx + bw / 2,
            0,
            IMAGE_WIDTH,
        )

        ymax = np.clip(
            cy + bh / 2,
            0,
            IMAGE_HEIGHT,
        )

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([
            float(xmin),
            float(ymin),
            float(xmax),
            float(ymax),
            confidence,
        ])

    return nms(
        boxes,
        NMS_THRESHOLD,
    )


# ============================================================
# Session
# ============================================================

def create_session(model_path):
    return ort.InferenceSession(
        str(model_path),
        providers=[
            "CPUExecutionProvider"
        ],
    )


print("=" * 72)
print("FP32 vs INT8 PERSON DETECTOR CAMERA COMPARISON")
print("=" * 72)

fp32_session = create_session(
    FP32_MODEL
)

int8_session = create_session(
    INT8_MODEL
)

fp32_input = fp32_session.get_inputs()[0].name
int8_input = int8_session.get_inputs()[0].name

print(f"FP32 : {FP32_MODEL.name}")
print(f"INT8 : {INT8_MODEL.name}")

print()
print(
    "FP32 providers:",
    fp32_session.get_providers(),
)

print(
    "INT8 providers:",
    int8_session.get_providers(),
)


# ============================================================
# Initial warmup
# ============================================================

dummy = np.zeros(
    (
        1,
        3,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ),
    dtype=np.float32,
)


for _ in range(5):
    fp32_session.run(
        None,
        {
            fp32_input: dummy
        },
    )

    int8_session.run(
        None,
        {
            int8_input: dummy
        },
    )


# ============================================================
# Camera
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW,
)

if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Camera could not be opened."
    )


print()
print("Camera started.")
print("Q : finish and print benchmark")
print("[ : threshold -0.05")
print("] : threshold +0.05")
print()


threshold = CONF_THRESHOLD

frame_index = 0

fp32_times = []
int8_times = []

fp32_detection_counts = []
int8_detection_counts = []

confidence_differences = []
bbox_ious = []


# ============================================================
# Main Loop
# ============================================================

while True:
    success, frame = cap.read()

    if not success:
        print("Camera read failed.")
        break

    frame_index += 1

    original_h, original_w = (
        frame.shape[:2]
    )

    input_tensor = preprocess(
        frame
    )


    # ========================================================
    # FP32
    # ========================================================

    start = time.perf_counter()

    fp32_output = fp32_session.run(
        None,
        {
            fp32_input:
            input_tensor
        },
    )[0]

    fp32_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    # ========================================================
    # INT8
    # ========================================================

    start = time.perf_counter()

    int8_output = int8_session.run(
        None,
        {
            int8_input:
            input_tensor
        },
    )[0]

    int8_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    # ========================================================
    # Decode
    # ========================================================

    fp32_boxes = decode_prediction(
        fp32_output[0],
        threshold,
    )

    int8_boxes = decode_prediction(
        int8_output[0],
        threshold,
    )


    # ========================================================
    # Benchmark recording
    #
    # 초기 프레임은 ORT / CPU warmup 영향을 받을 수 있으므로 제외
    # ========================================================

    if frame_index > WARMUP_FRAMES:

        fp32_times.append(
            fp32_ms
        )

        int8_times.append(
            int8_ms
        )

        fp32_detection_counts.append(
            len(fp32_boxes)
        )

        int8_detection_counts.append(
            len(int8_boxes)
        )


        # 가장 높은 confidence detection끼리 비교
        if fp32_boxes and int8_boxes:

            fp32_best = max(
                fp32_boxes,
                key=lambda x: x[4],
            )

            int8_best = max(
                int8_boxes,
                key=lambda x: x[4],
            )

            confidence_differences.append(
                abs(
                    fp32_best[4]
                    - int8_best[4]
                )
            )

            bbox_ious.append(
                calculate_iou(
                    fp32_best[:4],
                    int8_best[:4],
                )
            )


    # ========================================================
    # Drawing
    # ========================================================

    display = frame.copy()

    scale_x = (
        original_w / IMAGE_WIDTH
    )

    scale_y = (
        original_h / IMAGE_HEIGHT
    )


    # FP32 = green
    for box in fp32_boxes:
        xmin, ymin, xmax, ymax, conf = box

        x1 = int(xmin * scale_x)
        y1 = int(ymin * scale_y)
        x2 = int(xmax * scale_x)
        y2 = int(ymax * scale_y)

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"FP32 {conf:.2f}",
            (
                x1,
                max(20, y1 - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )


    # INT8 = orange/red-ish
    for box in int8_boxes:
        xmin, ymin, xmax, ymax, conf = box

        x1 = int(xmin * scale_x)
        y1 = int(ymin * scale_y)
        x2 = int(xmax * scale_x)
        y2 = int(ymax * scale_y)

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 140, 255),
            2,
        )

        cv2.putText(
            display,
            f"INT8 {conf:.2f}",
            (
                x1,
                min(
                    original_h - 10,
                    y2 + 20,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 140, 255),
            2,
        )


    # ========================================================
    # Overlay
    # ========================================================

    info = [
        f"Threshold: {threshold:.2f}",
        (
            f"FP32: {fp32_ms:.1f} ms "
            f"/ persons {len(fp32_boxes)}"
        ),
        (
            f"INT8: {int8_ms:.1f} ms "
            f"/ persons {len(int8_boxes)}"
        ),
    ]


    if fp32_ms > 0 and int8_ms > 0:

        speedup = (
            fp32_ms / int8_ms
        )

        info.append(
            f"INT8 speedup: {speedup:.2f}x"
        )


    for index, text in enumerate(info):

        cv2.putText(
            display,
            text,
            (
                10,
                25 + index * 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )


    cv2.imshow(
        "FP32 Green / INT8 Orange",
        display,
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


    elif key == ord("["):

        threshold = max(
            0.05,
            threshold - 0.05,
        )

        print(
            f"Threshold: {threshold:.2f}"
        )


    elif key == ord("]"):

        threshold = min(
            0.95,
            threshold + 0.05,
        )

        print(
            f"Threshold: {threshold:.2f}"
        )


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()


# ============================================================
# Summary
# ============================================================

print()
print("=" * 72)
print("BENCHMARK RESULT")
print("=" * 72)


if not fp32_times:
    print(
        "Not enough frames were collected."
    )
    raise SystemExit


fp32_times = np.array(
    fp32_times,
    dtype=np.float64,
)

int8_times = np.array(
    int8_times,
    dtype=np.float64,
)


fp32_avg = float(
    np.mean(fp32_times)
)

int8_avg = float(
    np.mean(int8_times)
)

fp32_median = float(
    np.median(fp32_times)
)

int8_median = float(
    np.median(int8_times)
)

fp32_p95 = float(
    np.percentile(
        fp32_times,
        95,
    )
)

int8_p95 = float(
    np.percentile(
        int8_times,
        95,
    )
)


print(
    f"Measured frames : "
    f"{len(fp32_times)}"
)

print()

print(
    f"FP32 average    : "
    f"{fp32_avg:.2f} ms"
)

print(
    f"FP32 median     : "
    f"{fp32_median:.2f} ms"
)

print(
    f"FP32 P95        : "
    f"{fp32_p95:.2f} ms"
)

print(
    f"FP32 infer FPS  : "
    f"{1000.0 / fp32_avg:.2f}"
)


print()

print(
    f"INT8 average    : "
    f"{int8_avg:.2f} ms"
)

print(
    f"INT8 median     : "
    f"{int8_median:.2f} ms"
)

print(
    f"INT8 P95        : "
    f"{int8_p95:.2f} ms"
)

print(
    f"INT8 infer FPS  : "
    f"{1000.0 / int8_avg:.2f}"
)


print()

print(
    f"INT8 speedup    : "
    f"{fp32_avg / int8_avg:.2f}x"
)


fp32_mean_detections = float(
    np.mean(
        fp32_detection_counts
    )
)

int8_mean_detections = float(
    np.mean(
        int8_detection_counts
    )
)


print()

print(
    f"FP32 avg persons: "
    f"{fp32_mean_detections:.3f}"
)

print(
    f"INT8 avg persons: "
    f"{int8_mean_detections:.3f}"
)


if confidence_differences:

    print()

    print(
        "Mean confidence difference : "
        f"{np.mean(confidence_differences):.4f}"
    )


if bbox_ious:

    print(
        "Mean FP32/INT8 bbox IoU    : "
        f"{np.mean(bbox_ious):.4f}"
    )


print()

print(
    "FP32 file size : "
    f"{FP32_MODEL.stat().st_size / 1024 / 1024:.2f} MB"
)

print(
    "INT8 file size : "
    f"{INT8_MODEL.stat().st_size / 1024 / 1024:.2f} MB"
)


print("=" * 72)