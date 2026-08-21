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

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "person_detector_fp32.onnx"


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


def nms(boxes, iou_threshold):
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

        remaining = []

        for box in boxes:
            if calculate_iou(
                best[:4],
                box[:4]
            ) < iou_threshold:
                remaining.append(box)

        boxes = remaining

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

    image = image.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1),
    )

    # CHW -> NCHW
    image = np.expand_dims(
        image,
        axis=0,
    )

    return np.ascontiguousarray(
        image,
        dtype=np.float32,
    )


# ============================================================
# Postprocess
# ============================================================

def decode_prediction(
    prediction,
    conf_threshold,
):
    # prediction:
    # [5, 15, 20]

    grid_h = prediction.shape[1]
    grid_w = prediction.shape[2]

    cell_w = IMAGE_WIDTH / grid_w
    cell_h = IMAGE_HEIGHT / grid_h

    objectness = sigmoid(
        prediction[0]
    )

    ys, xs = np.where(
        objectness >= conf_threshold
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

        # 학습 코드와 일치시키기 위해
        # width / height 역시 sigmoid 사용
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

        xmin = cx - bw / 2.0
        ymin = cy - bh / 2.0

        xmax = cx + bw / 2.0
        ymax = cy + bh / 2.0

        xmin = np.clip(
            xmin,
            0,
            IMAGE_WIDTH,
        )

        ymin = np.clip(
            ymin,
            0,
            IMAGE_HEIGHT,
        )

        xmax = np.clip(
            xmax,
            0,
            IMAGE_WIDTH,
        )

        ymax = np.clip(
            ymax,
            0,
            IMAGE_HEIGHT,
        )

        if xmax <= xmin:
            continue

        if ymax <= ymin:
            continue

        boxes.append(
            [
                float(xmin),
                float(ymin),
                float(xmax),
                float(ymax),
                confidence,
            ]
        )

    return boxes


# ============================================================
# ONNX Runtime
# ============================================================

print("=" * 65)
print("PERSON DETECTOR - ONNX FP32 CAMERA TEST")
print("=" * 65)

print(f"Model: {MODEL_PATH}")


session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)


input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

input_name = input_info.name
output_name = output_info.name


print()
print("ONNX Runtime")
print(
    f"Provider     : "
    f"{session.get_providers()}"
)
print(
    f"Input name   : {input_name}"
)
print(
    f"Input shape  : {input_info.shape}"
)
print(
    f"Input type   : {input_info.type}"
)
print(
    f"Output name  : {output_name}"
)
print(
    f"Output shape : {output_info.shape}"
)


# ============================================================
# Dummy verification
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


dummy_output = session.run(
    [output_name],
    {
        input_name: dummy
    },
)[0]


print()
print(
    "Runtime output shape:",
    dummy_output.shape,
)


if dummy_output.shape != (
    1,
    5,
    15,
    20,
):
    raise RuntimeError(
        "Unexpected ONNX output shape."
    )


print("Shape check: PASS")
print("=" * 65)


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
        "Camera 0 could not be opened."
    )


print()
print("Camera started.")
print("Q : quit")
print("[ : confidence -0.05")
print("] : confidence +0.05")
print()


current_threshold = CONF_THRESHOLD

fps_smoothed = 0.0

frame_count = 0


# ============================================================
# Camera Loop
# ============================================================

while True:
    total_start = time.perf_counter()


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:
        print(
            "Camera frame read failed."
        )
        break


    original_h, original_w = (
        frame.shape[:2]
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    start = time.perf_counter()

    input_tensor = preprocess(
        frame
    )

    preprocess_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    # --------------------------------------------------------
    # ONNX Inference
    # --------------------------------------------------------

    start = time.perf_counter()

    output = session.run(
        [output_name],
        {
            input_name:
            input_tensor
        },
    )[0]

    inference_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    # --------------------------------------------------------
    # Postprocess
    # --------------------------------------------------------

    start = time.perf_counter()

    boxes = decode_prediction(
        output[0],
        current_threshold,
    )

    boxes = nms(
        boxes,
        NMS_THRESHOLD,
    )

    postprocess_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    scale_x = (
        original_w
        / IMAGE_WIDTH
    )

    scale_y = (
        original_h
        / IMAGE_HEIGHT
    )


    for box in boxes:
        xmin, ymin, xmax, ymax, confidence = box

        x1 = int(
            xmin * scale_x
        )

        y1 = int(
            ymin * scale_y
        )

        x2 = int(
            xmax * scale_x
        )

        y2 = int(
            ymax * scale_y
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )


        # 사람의 bbox 중심
        center_x = (
            x1 + x2
        ) // 2

        center_y = (
            y1 + y2
        ) // 2


        # 캘리브레이션용 후보 좌표:
        # 사람 bbox의 하단 중앙
        bottom_x = center_x
        bottom_y = y2


        cv2.circle(
            frame,
            (
                bottom_x,
                bottom_y,
            ),
            5,
            (0, 0, 255),
            -1,
        )


        cv2.putText(
            frame,
            f"Person {confidence:.2f}",
            (
                x1,
                max(
                    20,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000.0


    fps = (
        1000.0 / total_ms
        if total_ms > 0
        else 0.0
    )


    if fps_smoothed == 0:
        fps_smoothed = fps

    else:
        fps_smoothed = (
            fps_smoothed * 0.9
            + fps * 0.1
        )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    lines = [
        "ONNX FP32 / CPU",
        (
            f"Threshold: "
            f"{current_threshold:.2f}"
        ),
        (
            f"Persons: "
            f"{len(boxes)}"
        ),
        (
            f"Preprocess: "
            f"{preprocess_ms:.1f} ms"
        ),
        (
            f"Inference: "
            f"{inference_ms:.1f} ms"
        ),
        (
            f"Postprocess: "
            f"{postprocess_ms:.1f} ms"
        ),
        (
            f"FPS: "
            f"{fps_smoothed:.2f}"
        ),
    ]


    for index, text in enumerate(
        lines
    ):
        cv2.putText(
            frame,
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


    # --------------------------------------------------------
    # Periodic terminal report
    # --------------------------------------------------------

    frame_count += 1


    if frame_count % 60 == 0:
        print(
            f"persons={len(boxes)} | "
            f"threshold="
            f"{current_threshold:.2f} | "
            f"pre="
            f"{preprocess_ms:.1f} ms | "
            f"inference="
            f"{inference_ms:.1f} ms | "
            f"post="
            f"{postprocess_ms:.1f} ms | "
            f"FPS="
            f"{fps_smoothed:.2f}"
        )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "Person Detector - ONNX FP32",
        frame,
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


    elif key == ord("["):
        current_threshold = max(
            0.05,
            current_threshold - 0.05,
        )

        print(
            "Confidence threshold:",
            f"{current_threshold:.2f}",
        )


    elif key == ord("]"):
        current_threshold = min(
            0.95,
            current_threshold + 0.05,
        )

        print(
            "Confidence threshold:",
            f"{current_threshold:.2f}",
        )


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("ONNX FP32 camera test finished.")