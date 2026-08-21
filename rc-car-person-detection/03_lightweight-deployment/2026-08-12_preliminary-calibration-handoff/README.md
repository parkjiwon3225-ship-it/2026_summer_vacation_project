# 2026-08-12 Calibration Handoff — Preliminary/Legacy

> 최종 모델 전 팀 간 ONNX 연결을 확인하기 위해 사용한 임시 전달본입니다. 현재 배포 후보는 같은 상위 파트의 `2026-08-13_results4-pi-model-variants`를 사용하세요.

## 1. 목적

이 폴더는 RC-Car Person Detection 프로젝트의 **최종 딥러닝 모델 완성 전**, 캘리브레이션/위치 계산 팀의 개발과 테스트를 진행하기 위한 임시 모델 전달본입니다.

현재 모델은 최종 모델이 아니며, 이번 테스트의 목적은 다음과 같습니다.

- ONNX 모델이 실제 개발 파이프라인에서 정상 동작하는지 확인
- 카메라 프레임 → Person Detection → Bounding Box 전달 구조 확인
- FP32와 INT8 경량화 모델의 차이 확인
- 향후 최종 딥러닝 모델 완성 후 사용할 배포/경량화 방향 검토

최종 Person Detector가 완성되면 모델 변환 및 경량화 테스트는 다시 진행할 예정입니다.

---

## 2. 파일 구성

### `person_detector_fp32.onnx`

기준 모델입니다.

- 원본 PyTorch 학습 결과에서 직접 ONNX FP32로 변환
- PyTorch와 ONNX Runtime 출력 일치 검증 완료
- 현재 캘리브레이션 개발에서는 **이 모델을 기준으로 사용하는 것을 권장**

### `person_detector_int8.onnx`

FP32 ONNX를 Static INT8 Quantization한 실험 모델입니다.

- Calibration image 26장 사용
- 경량화 및 속도 차이 확인용
- 일부 새로운 장면에서 FP32와 detection/bbox 차이가 확인됨
- 최종 모델이 아닌 **INT8 경량화 실험용**

### `test_fp32_camera.py`

OpenCV 카메라 입력과 ONNX Runtime을 이용한 FP32 모델 실시간 테스트 코드입니다.

### `compare_fp32_int8_camera.py`

동일 카메라 프레임을 FP32와 INT8 모델에 입력하여 결과와 추론 속도를 비교하는 코드입니다.

### `benchmark_results.csv`

노트북 CPU에서 수행한 독립 inference benchmark 결과입니다.

---

## 3. Model Input

```text
Shape  : [1, 3, 240, 320]
Layout : NCHW
Type   : float32
Color  : RGB
Range  : 0.0 ~ 1.0
```

전처리:

```text
OpenCV BGR
    ↓
BGR → RGB
    ↓
Resize 320 × 240
    ↓
float32 / 255.0
    ↓
HWC → CHW
    ↓
Batch dimension 추가
    ↓
[1, 3, 240, 320]
```

별도의 ImageNet mean/std normalization은 사용하지 않습니다.

---

## 4. Model Output

```text
Shape : [1, 5, 15, 20]
```

채널:

```text
0 : objectness logit
1 : center X offset logit
2 : center Y offset logit
3 : bounding box width logit
4 : bounding box height logit
```

후처리 기준:

```python
confidence = sigmoid(output[0])

cx = (grid_x + sigmoid(output[1])) * cell_width
cy = (grid_y + sigmoid(output[2])) * cell_height

width = sigmoid(output[3]) * 320
height = sigmoid(output[4]) * 240
```

현재 output grid는 `20 × 15`이며 cell 크기는 `16 × 16 px`입니다.

---

## 5. Detection 좌표

Bounding Box:

```text
xmin, ymin, xmax, ymax
```

중심점:

```text
center_x = (xmin + xmax) / 2
center_y = (ymin + ymax) / 2
```

사람의 지면 위치 계산 등에 사용할 수 있는 후보 좌표:

```text
bottom_center_x = (xmin + xmax) / 2
bottom_center_y = ymax
```

`bottom-center` 사용 여부는 캘리브레이션 결과에 따라 최종 결정합니다.

---

## 6. 현재 테스트 설정

```text
Confidence threshold : 0.20
NMS IoU threshold    : 0.20
```

현재 임시 모델용 설정이며 최종 모델에서는 다시 결정합니다.

---

## 7. 모델 크기 및 노트북 Benchmark

| Model | Size | Average Inference | Median | P95 |
|---|---:|---:|---:|---:|
| ONNX FP32 | 10.94 MB | 44.81 ms | 45.85 ms | 55.15 ms |
| ONNX INT8 | 2.82 MB | 2.60 ms | 2.59 ms | 2.87 ms |

INT8 모델은 FP32 ONNX 대비 파일 크기가 약 **74% 감소**했습니다.

위 속도는 노트북 CPU에서 측정한 **순수 모델 inference 시간**입니다.

카메라 capture, preprocessing, postprocessing, 통신 등을 포함한 전체 시스템 FPS가 아니며, **Raspberry Pi 4에서 동일한 성능을 보장하지 않습니다.**

---

## 8. INT8 품질 테스트

INT8 calibration에 사용하지 않은 새로운 카메라 프레임 100개로 FP32와 INT8을 비교했습니다.

```text
Frames evaluated           : 100
Detection count agreement  : 92.00%
FP32 detections            : 106
INT8 detections            : 100
Mean confidence difference : 0.008965
Mean bbox IoU              : 0.949144
```

추가 문제 프레임 분석에서는 일부 detection count 및 Bounding Box 위치 차이가 확인되었습니다.

따라서 현재 권장 방식은:

```text
FP32 ONNX
→ 캘리브레이션 개발 기준

INT8 ONNX
→ 경량화가 속도 및 bbox 좌표에 미치는 영향 확인용
```

입니다.

---

## 9. 실행 환경

필요 패키지:

```bash
pip install numpy opencv-python onnxruntime
```

FP32 카메라 테스트:

```bash
python test_fp32_camera.py
```

FP32 / INT8 비교:

```bash
python compare_fp32_int8_camera.py
```

---

## 10. 주의사항

이 모델은 **최종 Person Detection 모델이 아닙니다.**

현재 프로젝트의 우선 작업은 최종 딥러닝 모델의 학습 및 정확도 개선입니다.

따라서 현재 결과만으로 다음 사항을 확정하지 않습니다.

- 최종 Raspberry Pi FPS
- 최종 INT8 사용 여부
- 최종 confidence threshold
- 최종 Bounding Box 정확도
- 최종 모델 구조
- 최종 위치 계산 방식

향후 최종 딥러닝 모델이 완성되면 다음 과정을 다시 수행합니다.

```text
Final Person Detector
        ↓
ONNX FP32
        ↓
Raspberry Pi benchmark
        ↓
INT8 등 경량화 비교
        ↓
Detection / bbox / FPS 검증
        ↓
최종 배포 방식 결정
```
