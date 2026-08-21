# 2026-08-18 Results27 ONNX / Selective INT8 경량화

## 목적

현재 accuracy champion인 results27 640×480 모델을 Raspberry Pi 4에서 사용할 수 있도록 ONNX 변환과 INT8 경량화를 수행했다.

기준 모델:

- Run: longrun_s5_stage1_640_seed11
- Input: 640×480
- Best epoch: 52
- mAP50:95: 0.2880503189

---

## FP32 ONNX Reference

파일:

`models/results27_640_fp32.onnx`

PyTorch best.pt를 FP32 ONNX로 변환했다.

출력 shape:

- boxes: [1, 25500, 4]
- scores: [1, 25500]

PyTorch와 ONNX Runtime 실영상 비교:

- score max abs error: 약 5.96e-7
- box max abs error: 약 2.98e-4

따라서 FP32 ONNX는 PyTorch 원본과 사실상 동일한 reference 모델로 사용한다.

---

## INT8 Round 1

일반적인 PTQ 방식을 먼저 시험했다.

시험 모델:

- Conv MinMax
- Conv Percentile
- Full MinMax

30개 validation image 기준:

- Conv MinMax retain@0.17: 약 59.83%
- Conv Percentile retain@0.17: 약 78.17%
- Full MinMax retain@0.17: 약 61.57%

결론:

전체 또는 단순 Conv INT8 양자화는 FP32 출력 보존성이 부족해 탈락했다.

---

## INT8 Round 2

dtype와 calibration 조합을 변경했다.

시험:

- S8S8 Entropy
- U8S8 MinMax
- U8U8 MinMax

60개 validation image 기준:

- S8S8 Entropy retain@0.17: 약 60.95%
- U8S8 MinMax retain@0.17: 약 60.65%
- U8U8 MinMax retain@0.17: 약 62.72%

결론:

dtype와 calibration 방식을 바꾸는 것만으로는 results27을 안정적으로 INT8화하지 못했다.

---

## INT8 Round 3

전체 양자화 대신 selective quantization을 시작했다.

시험:

- backbone only
- backbone + FPN
- no head

80개 validation image 기준:

- backbone only retain@0.17: 약 80.04%
- backbone + FPN retain@0.17: 약 79.68%
- no head retain@0.17: 약 79.68%

Round 1과 Round 2보다는 개선됐지만 최종 배포 수준에는 부족했다.

이 결과를 통해 backbone 내부에서도 PTQ 민감도가 다를 가능성이 있다고 판단했다.

---

## INT8 Round 4 - Backbone Sensitivity

Backbone Conv node를 순서대로 Q1, Q2, Q3, Q4로 나눠 부분 양자화했다.

### Q1 early only

- retain@0.17: 약 78.25%

초기 backbone은 PTQ에 매우 민감했다.

### Q2 only

- retain@0.17: 약 97.42%
- IoU: 약 0.9826

### Q3 only

- retain@0.17: 약 98.14%
- IoU: 약 0.9775

### Q4 only

- retain@0.17: 약 97.85%
- IoU: 약 0.9780

### Q3 + Q4

파일:

`models/results27_640_int8_round4_q3_q4_suffix50.onnx`

주요 결과:

- file size: 약 0.73 MB
- retain@0.15: 약 96.35%
- retain@0.17: 약 97.42%
- retain@0.20: 약 96.47%
- retain@0.25: 약 95.79%

핵심 발견:

초기 backbone Q1은 FP32로 유지하는 것이 중요하고, 후반 backbone은 비교적 안전하게 INT8로 변환할 수 있었다.

---

## INT8 Round 5 - Final Mix

추가 시험:

- Q2 + Q3
- Q2 + Q4
- Q2 + Q3 + Q4
- Round4 Q3 + Q4 재검증

120개 validation image 기준 Q3 + Q4:

- size: 약 0.73 MB
- retain@0.17: 약 96.19%
- IoU: 약 0.9704

Q2 + Q3 + Q4:

- size: 약 0.69 MB
- retain@0.17: 약 96.08%

Q2까지 INT8 범위를 넓히면 파일은 약간 더 작아지지만:

- latency 증가
- extra detection 증가
- IoU 소폭 감소

가 발생했다.

따라서 현재 results27 selective INT8 1순위는:

`results27_640_int8_round4_q3_q4_suffix50.onnx`

이다.

구성:

- Early backbone: FP32
- Q3 + Q4: INT8
- FPN: FP32
- Detection head: FP32

---

## 파일 크기

대략적인 비교:

- results27 FP32: 약 1.39 MB
- results27 Q3+Q4 Mixed INT8: 약 0.76 MB

약 45% 수준의 파일 크기 감소를 얻었다.

---

## PC 4모델 카메라 비교

비교 모델:

1. results20 FP32 320×240
2. results20 INT8 Percentile 320×240
3. results27 FP32 640×480
4. results27 Q3+Q4 Mixed INT8 640×480

### Confidence 0.17 / 300 frames

results20 FP32:

- infer: 약 5.46 ms
- detections: 650

results20 INT8 Percentile:

- infer: 약 10.24 ms
- detections: 1087

results27 FP32:

- infer: 약 21.30 ms
- detections: 354

results27 Mixed Q3+Q4:

- infer: 약 23.69 ms
- detections: 353

results20 INT8 Percentile은 동일 results20 FP32보다 detection 수가 크게 증가했다.

반면 results27은 FP32 354, Mixed 353으로 매우 유사했다.

### Confidence 0.20 / 300 frames

results20 FP32:

- infer: 약 5.70 ms
- detections: 344

results20 INT8 Percentile:

- infer: 약 10.75 ms
- detections: 369

results27 FP32:

- infer: 약 21.18 ms
- detections: 272

results27 Mixed Q3+Q4:

- infer: 약 24.48 ms
- detections: 271

results27 FP32와 Mixed INT8의 출력 보존성이 실제 카메라에서도 안정적인 편이었다.

---

## Raspberry Pi 실제 통합 상태

2026-08-18 현재 Raspberry Pi 4에서 실제 전체 시스템 통합 시험에 사용한 모델은:

`results20_fpn48_int8_qdq_percentile.onnx`

이다.

카메라, 사람 탐지, RC카 관련 기능을 함께 실행한 상태에서:

- 약 21~27 FPS
- 시스템 과부하 없음
- 다른 기능과 동시 실행 가능
- 사람 탐지는 실용 가능한 수준
- 일부 false positive / false negative 존재

로 확인됐다.

주의:

위 21~27 FPS는 results27 Mixed 모델의 수치가 아니다.

results27 FP32와 results27 Q3+Q4 Mixed는 Raspberry Pi에서 별도 실측이 필요하다.

---

## 현재 배포 후보

기존 Raspberry Pi 기준:

`results20_fpn48_int8_qdq_percentile.onnx`

새 accuracy reference:

`results27_640_fp32.onnx`

새 selective INT8 후보:

`results27_640_int8_round4_q3_q4_suffix50.onnx`

PC x86 환경에서는 INT8 모델이 FP32보다 빠르지 않았다.

따라서 최종 배포 모델은 Raspberry Pi ARM 환경에서 직접 비교한다.

평가 항목:

- inference latency
- end-to-end FPS
- CPU usage
- RAM usage
- temperature
- 실제 운동장 false positive
- 실제 운동장 false negative
- 작은 사람 및 먼 사람 탐지 성능

---

## 현재 최적화 방향

기존 results20 INT8 모델이 Raspberry Pi 전체 기능 동시 실행 상태에서도 약 21~27 FPS를 확보했다.

따라서 앞으로의 목표는 최대 FPS 자체보다는:

실시간성을 유지하면서 정확도를 높이는 것

으로 이동한다.

해상도 증가 모델과 selective INT8 모델을 Pi에서 비교해 accuracy-FPS 최적점을 찾는다.

---

## 운동장 Hard Negative 계획

운동장 전체 시험 중 특정 사물을 반복적으로 사람으로 인식하는 경우 hard-negative 데이터 추가를 검토한다.

예:

- 골대
- 나무
- 기둥
- 가방
- 표지판
- 기타 반복적으로 오검출되는 구조물

한두 번 순간적으로 발생하는 FP보다 여러 프레임과 여러 각도에서 반복되는 FP를 우선 수집한다.

사람이 포함된 이미지를 annotation 없이 negative image로 사용하는 것은 피한다.

---

## 주요 파일

FP32 reference:

`models/results27_640_fp32.onnx`

현재 selective INT8 1순위:

`models/results27_640_int8_round4_q3_q4_suffix50.onnx`

경량화 검증 결과:

`results/`

재현 스크립트:

`scripts/`

results20 / results27 비교:

`comparison/`
