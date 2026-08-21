# Person Detector TAIL004 — 최종 릴리스

## 바로 사용할 파일

[`models/person_detector_tail004_int8.onnx`](models/person_detector_tail004_int8.onnx)

SHA-256:

```text
230755c15376065bdfbcea44cbc9259d9691ec48319bc96bdbfb15c38b3e01be
```

FP32 회귀 비교가 필요하면 [`models/person_detector_r46_fp32_reference.onnx`](models/person_detector_r46_fp32_reference.onnx)을 사용한다.

## 고정 사양

| 항목 | 값 |
|---|---|
| Source | R46 `r46_final448_seed15_100e`, epoch 25 |
| Architecture | DSConv+Residual backbone, FPN48, anchor-free head |
| Model input | `images`, float32 `[1,3,336,448]`, RGB, 0~1 |
| Model output | `boxes` `[1,12502,4]`, `scores` `[1,12502]` |
| Quantization | QDQ selective INT8, Q2 tail4 + Q3 + Q4 |
| Starting threshold | confidence 0.25, NMS IoU 0.5 |
| Runtime | Raspberry Pi, ONNX Runtime C++ CPU |
| Camera input | OpenCV C++ 320×240 `CV_8UC3` BGR |

## 실행 파이프라인

```text
OpenCV C++ camera (320×240 BGR)
  → capture timestamp / frame_id
  → BGR→RGB
  → 448×336 letterbox, float32 / 255, NCHW
  → ONNX Runtime C++
  → confidence filter 0.25
  → NMS 0.5
  → original 320×240 coordinates
  → persons[] 전달
```

ONNX Runtime은 카메라를 열지 않는다. OpenCV가 프레임을 획득하고 ONNX Runtime은 모델 추론만 담당한다.

구현 세부사항은 [`INTEGRATION_CONTRACT.md`](INTEGRATION_CONTRACT.md), 정확도와 한계는 [`MODEL_CARD.md`](MODEL_CARD.md)를 확인한다.

## 최종 선택 근거

- R46은 448×336 후보 중 Recall 0.412851, small recall 0.155005로 가장 균형이 좋았다.
- TAIL004는 FP32 detection retain 97.90%, mean IoU 0.9746을 유지했다.
- FP32 대비 모델 크기를 47.25% 줄였다.
- Q3Q4 control보다 4개 Conv를 더 INT8로 전환하면서 보존율을 유지했다.
- TAIL006보다 양자화 범위를 보수적으로 제한하고 라즈베리파이 실기 안정성을 우선했다.

오프라인 수치와 전체 비교는 `03_lightweight-deployment/2026-08-20_r46-tail004-final/`에 보존되어 있다.
