"""
================================================================================
Person Detector v2 — OpenCV 실시간 카메라 감지 (Anchor 기반 모델용)
================================================================================

⚠️ v1용 camera_detection_final.py와는 호환되지 않습니다. v2는 anchor 5개를
쓰기 때문에 모델 출력 구조 자체가 다릅니다([5,H,W] → [anchor수,5,H,W]).

필요한 파일 2개 (같은 폴더에 둘 다 있어야 함):
  1) person_detector_v2_script.pt   (또는 학습 중 자동 저장된 best_script.pt)
  2) anchors.json                    (또는 best_anchors.json) — anchor 크기 등 메타정보

사용법:
  python camera_detection_v2.py --model person_detector_v2_script.pt --anchors anchors.json
  python camera_detection_v2.py --model person_detector_v2_script.pt --anchors anchors.json --conf 0.3

단축키: q=종료, s=스크린샷, r=녹화 시작/중지
================================================================================
"""

import cv2
import torch
import numpy as np
import time
import argparse
import os
import sys
import json
from collections import deque


# ============================================================================
# [1부] 명령줄 인자
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Person Detector v2 (Anchor 기반, TorchScript)")
    parser.add_argument("--model", type=str, default="person_detector_v2_script.pt",
                         help="TorchScript(.pt) 모델 경로")
    parser.add_argument("--anchors", type=str, default="anchors.json",
                         help="anchor 정보 JSON 경로 (학습 노트북에서 함께 저장됨)")
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    parser.add_argument("--width", type=int, default=1920, help="카메라 요청 해상도(가로)")
    parser.add_argument("--height", type=int, default=1080, help="카메라 요청 해상도(세로)")
    parser.add_argument("--conf", type=float, default=None,
                         help="confidence 임계값 (지정 안하면 anchors.json의 값을 사용)")
    parser.add_argument("--nms", type=float, default=None,
                         help="NMS IoU 임계값 (지정 안하면 anchors.json의 값을 사용)")
    parser.add_argument("--min-area", type=float, default=0.0005, help="박스 최소 면적 비율")
    parser.add_argument("--max-area", type=float, default=0.9, help="박스 최대 면적 비율")
    parser.add_argument("--record", type=str, default=None, help="저장할 mp4 파일 경로")
    parser.add_argument("--cpu", action="store_true", help="GPU가 있어도 CPU로 강제 실행")
    return parser.parse_args()


args = parse_args()

DEVICE = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

print("=" * 70)
print("Person Detector v2 - OpenCV 카메라 감지 시스템 (Anchor 기반)")
print("=" * 70)
print(f"사용 장치: {DEVICE.upper()}")


# ============================================================================
# [2부] anchor 설정 로드
#
# v1과의 가장 큰 차이: 모델 출력을 박스로 바꾸려면 "anchor 크기"가
# 반드시 필요합니다. 이건 모델(.pt) 안에는 없고, 학습 때 따로 저장한
# JSON 파일에만 있습니다. 그래서 두 파일을 항상 같이 들고 다녀야 합니다.
# ============================================================================
if not os.path.isfile(args.anchors):
    print(f"❌ [오류] anchor 설정 파일을 찾을 수 없습니다: {args.anchors}")
    print("   학습 노트북의 results/<실험이름>/anchors.json")
    print("   또는 checkpoints/<실험이름>/person_detector_v2_best_anchors.json 을 복사해오세요.")
    sys.exit(1)

with open(args.anchors, "r") as f:
    anchor_cfg = json.load(f)

ANCHORS_WH = torch.tensor(anchor_cfg["anchors_wh"], dtype=torch.float32).to(DEVICE)
IMAGE_WIDTH = anchor_cfg["image_width"]
IMAGE_HEIGHT = anchor_cfg["image_height"]
CONF_THRESHOLD = args.conf if args.conf is not None else anchor_cfg.get("conf_threshold", 0.4)
NMS_IOU_THRESHOLD = args.nms if args.nms is not None else anchor_cfg.get("nms_iou_threshold", 0.5)

print(f"모델 경로       : {args.model}")
print(f"Anchor 파일     : {args.anchors}")
print(f"Anchor 개수     : {len(ANCHORS_WH)}")
print(f"입력 이미지 크기: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
print(f"신뢰도 임계값   : {CONF_THRESHOLD} (학습 때 보정된 값)")
print(f"NMS 임계값      : {NMS_IOU_THRESHOLD}")
print("=" * 70)
print()


# ============================================================================
# [3부] 모델 로드
# ============================================================================
if not os.path.isfile(args.model):
    print(f"❌ [오류] 모델 파일을 찾을 수 없습니다: {args.model}")
    sys.exit(1)

try:
    print("[로딩] 모델 로드 중...")
    model = torch.jit.load(args.model, map_location=DEVICE)
    model.eval()
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ [오류] 모델 로드 실패: {e}")
    sys.exit(1)

print("[로딩] 워밍업 중...")
with torch.no_grad():
    dummy_input = torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=DEVICE)
    try:
        dummy_out = model(dummy_input)
        print(f"✅ 워밍업 완료 (출력 shape: {tuple(dummy_out.shape)})")
        # v2는 [1, anchor수, 5, H, W] 형태여야 정상입니다
        if dummy_out.dim() != 5:
            print(f"⚠️  경고: 출력 차원이 5가 아닙니다({dummy_out.dim()}). v1 모델을 잘못 넣으신 건 아닌지 확인하세요.")
    except Exception as e:
        print(f"⚠️  워밍업 실패 (무시하고 계속): {e}")

print()


# ============================================================================
# [4부] 유틸리티 함수
# ============================================================================

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, threshold):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    result = []
    while boxes:
        best = boxes.pop(0)
        result.append(best)
        boxes = [b for b in boxes if calculate_iou(best[:4], b[:4]) < threshold]
    return result


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    return img.to(DEVICE)


def decode_prediction_v2(pred, anchors_wh, conf_threshold):
    """
    v2 전용 디코딩 — anchor 기반.

    pred: [num_anchors, 5, H, W] (모델 출력, 배치 차원 제거된 상태)
          채널0=objectness, 1~4=(tx,ty,tw,th)

    v1과 다른 점: anchor마다 "기준이 되는 박스 크기"가 있고, 모델은
    "그 기준에서 얼마나 벗어나는지"만 예측합니다. 그래서 디코딩할 때
    anchor의 (width, height)를 반드시 곱해줘야 합니다.
    """
    A, _, grid_h, grid_w = pred.shape
    cell_w = IMAGE_WIDTH / grid_w
    cell_h = IMAGE_HEIGHT / grid_h

    obj = torch.sigmoid(pred[:, 0])              # [A, H, W]
    tx = torch.sigmoid(pred[:, 1])
    ty = torch.sigmoid(pred[:, 2])
    tw = pred[:, 3].clamp(max=6.0)
    th = pred[:, 4].clamp(max=6.0)

    mask_idx = torch.nonzero(obj > conf_threshold, as_tuple=False)  # [K, 3] -> (anchor_idx, y, x)
    if mask_idx.shape[0] == 0:
        return []

    a_idx = mask_idx[:, 0]
    ys = mask_idx[:, 1]
    xs = mask_idx[:, 2]

    anchor_w = anchors_wh[a_idx, 0]
    anchor_h = anchors_wh[a_idx, 1]

    cx = (xs.float() + tx[a_idx, ys, xs]) * cell_w
    cy = (ys.float() + ty[a_idx, ys, xs]) * cell_h
    bw = anchor_w * torch.exp(tw[a_idx, ys, xs])
    bh = anchor_h * torch.exp(th[a_idx, ys, xs])

    xmin = cx - bw / 2
    ymin = cy - bh / 2
    xmax = cx + bw / 2
    ymax = cy + bh / 2
    scores = obj[a_idx, ys, xs]

    boxes = []
    for i in range(len(a_idx)):
        boxes.append([
            xmin[i].item(), ymin[i].item(), xmax[i].item(), ymax[i].item(), scores[i].item()
        ])
    return boxes


def confidence_color(score):
    score = max(0.0, min(1.0, score))
    return (0, int(255 * score), int(255 * (1 - score)))  # BGR


# ============================================================================
# [5부] 카메라 초기화
# ============================================================================
print("[카메라] 초기화 중...")
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"❌ [오류] 카메라(index={args.camera})를 열 수 없습니다.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"✅ 카메라 초기화 완료 (해상도: {actual_width}x{actual_height}, FPS: {camera_fps:.1f})\n")

cv2.namedWindow("Person Detector v2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Person Detector v2", 1280, 720)

fps_history = deque(maxlen=20)
prev_time = time.time()

writer = None
recording = args.record is not None


def init_writer(path, w, h, fps=20.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


# ============================================================================
# [6부] 메인 루프
# ============================================================================
print("[시작] 실시간 감지 시작... (q=종료, s=스크린샷, r=녹화)\n")

try:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  프레임을 읽지 못했습니다.")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        if recording and writer is None:
            writer = init_writer(args.record, w, h, camera_fps)

        input_tensor = preprocess(frame)

        try:
            with torch.inference_mode():
                output = model(input_tensor)   # [1, A, 5, grid_h, grid_w]
        except Exception as e:
            print(f"❌ [추론 오류] {e}")
            continue

        pred = output[0].cpu()   # [A, 5, grid_h, grid_w]

        boxes = decode_prediction_v2(pred, ANCHORS_WH.cpu(), CONF_THRESHOLD)
        boxes = nms(boxes, NMS_IOU_THRESHOLD)

        frame_area = w * h
        filtered_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax, score = box
            xmin = int(xmin / IMAGE_WIDTH * w); xmax = int(xmax / IMAGE_WIDTH * w)
            ymin = int(ymin / IMAGE_HEIGHT * h); ymax = int(ymax / IMAGE_HEIGHT * h)
            xmin, xmax = max(0, min(xmin, w-1)), max(0, min(xmax, w-1))
            ymin, ymax = max(0, min(ymin, h-1)), max(0, min(ymax, h-1))

            box_area = max(0, xmax - xmin) * max(0, ymax - ymin)
            area_ratio = box_area / frame_area if frame_area > 0 else 0
            if area_ratio < args.min_area or area_ratio > args.max_area:
                continue
            filtered_boxes.append((xmin, ymin, xmax, ymax, score))

        count = len(filtered_boxes)
        for xmin, ymin, xmax, ymax, score in filtered_boxes:
            color = confidence_color(score)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"Person {score*100:.1f}%"
            label_y = ymin - 10 if ymin > 30 else ymin + 25
            (tw_, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (xmin-2, label_y-th_-5), (xmin+tw_+2, label_y+5), color, -1)
            cv2.putText(frame, label, (xmin, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        now = time.time()
        instant_fps = 1 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        fps_history.append(instant_fps)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(frame, f"Detected: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if recording:
            cv2.putText(frame, "REC", (w-120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            if writer is not None:
                writer.write(frame)

        cv2.imshow("Person Detector v2", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[종료] q 키 입력")
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 [저장] {filename}")
        elif key == ord('r'):
            recording = not recording
            if recording:
                print("[녹화] 시작")
            else:
                print("[녹화] 중지")
                if writer is not None:
                    writer.release(); writer = None

        if cv2.getWindowProperty("Person Detector v2", cv2.WND_PROP_VISIBLE) < 1:
            print("[종료] 창 닫기")
            break

finally:
    print("\n[정리] 자원 해제 중...")
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ 프로그램 종료")
