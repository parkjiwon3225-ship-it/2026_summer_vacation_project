# 2026-08-16 results20 Camera Validation

`results20_fpn48` FP32/INT8 모델을 실제 카메라 입력에서 비교·진단하기 위한 패키지입니다.

## 목적

- FP32 / INT8 MinMax / INT8 Percentile 실시간 검출 비교
- 동일 카메라 프레임 기반 모델 비교
- ONNX Runtime 추론 지연시간 측정
- FP32 ↔ INT8 출력 차이 진단
- Raspberry Pi 배포 전 이상 동작 확인

## 구조

```text
2026-08-16_results20-camera-validation/
├─ README.md
├─ requirements.txt
├─ detect_compare.py
├─ diagnose_models.py
├─ models/
│  ├─ results20_fpn48_fp32.onnx
│  ├─ results20_fpn48_int8_qdq_minmax.onnx
│  └─ results20_fpn48_int8_qdq_percentile.onnx
└─ results/
   └─ diagnostics/
      ├─ diagnostic_summary.csv
      └─ diagnostic_pairwise.csv
```

## 설치

```bash
pip install -r requirements.txt
```

## 실시간 카메라 비교

```bash
python detect_compare.py --webcam 0
```

모델 전환:

- `1`: FP32
- `2`: INT8 MinMax
- `3`: INT8 Percentile
- `s`: 화면 저장
- `q`: 종료

## 상세 진단

```bash
python diagnose_models.py --frames 300 --threads 2
```

## 2026-08-16 재검증 결과

### 1. 기존 300프레임 무인 장면 재해석

기존 저장 결과의 `FP32 detection = 0`은 오류가 아니라 **사람이 없는 장면에서의 정상 동작**으로 확인했습니다.

같은 무인 장면에서:

- FP32: detection 0
- INT8 MinMax: detection 273
- INT8 Percentile: detection 406

따라서 기존 결과는 FP32 이상이 아니라 **INT8 양자화 후보에서 False Positive가 증가한 현상**으로 해석합니다.

### 2. 최종 checkpoint 확정

최종 FP32 기준 checkpoint:

- run: `home_final_s1_continue_results14_to100`
- best epoch: 35
- validation mAP50:95: `0.2607381170353619`
- best validation loss: `1.5935413719465334`
- input: `1x3x240x320`
- FPN channels: 48
- backbone expansion: 2.0

### 3. PyTorch checkpoint ↔ FP32 ONNX 동일성

동일 웹캠 프레임으로 PyTorch `best.pt`와 `results20_fpn48_fp32.onnx` raw output을 직접 비교했습니다.

- score max abs diff: `3.2782554626464844e-07`
- score mean abs diff: `3.6651428558798216e-08`
- box max abs diff: `6.103515625e-05`
- box mean abs diff: `4.348644779383903e-06`

결론: **FP32 ONNX export는 최종 PyTorch checkpoint와 사실상 동일합니다.**

### 4. 카메라 입력 진단

초기 프레임은 자동 노출 안정화 전이라 거의 검게 캡처될 수 있었습니다.

OpenCV backend를 충분히 워밍업한 뒤 정상 입력을 확인했습니다.

- DEFAULT/MSMF: 640x480, 약 30 FPS
- 정상 프레임 예시 BGR min/max/mean: `10 / 213 / 61.25`
- 입력 tensor min/max/mean: `0.04706 / 0.83529 / 0.24070`

학습 전처리는 RGB, `/255`, 320x240 letterbox이며 진단 스크립트도 동일 방식을 사용합니다.

### 5. 사람 포함 300프레임 재진단

환경:

- Windows 11
- Python 3.12.9
- ONNX Runtime 1.28.0
- CPUExecutionProvider
- threads: 2
- confidence threshold: 0.25
- NMS IoU: 0.5

#### 성능

| Model | Infer Mean | P50 | P95 | P99 | Total | FPS | Mean Max Score | Avg Det |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 4.810 ms | 4.634 | 6.017 | 7.238 | 5.479 ms | 182.53 | 0.3106 | 1.080 |
| INT8 MinMax | 9.165 ms | 9.053 | 10.659 | 11.565 | 9.898 ms | 101.03 | 0.3030 | 1.700 |
| INT8 Percentile | 9.401 ms | 9.254 | 10.862 | 12.865 | 10.149 ms | 98.54 | 0.3050 | 1.680 |

현재 Windows CPU 환경에서는 **QDQ INT8이 FP32보다 느렸습니다.**
이 결과를 Raspberry Pi 성능으로 일반화하지 않으며 Pi 실측이 필요합니다.

#### FP32 대비 INT8 detection preservation

| Model | FP32 det | INT8 det | Matched | Missed | Extra | Retain | Mean IoU | Mean Score Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 MinMax | 324 | 510 | 273 | 51 | 237 | 84.26% | 0.8666 | -0.0106 |
| INT8 Percentile | 324 | 504 | 282 | 42 | 222 | 87.04% | 0.8862 | -0.0063 |

Percentile이 FP32 보존율과 IoU는 조금 더 좋았지만 두 INT8 모델 모두 FP32보다 extra detection이 많았습니다.

## 현재 판단

### Primary

**FP32 ONNX**

- Validation 기준 최고 모델
- PyTorch ↔ ONNX raw output 일치 확인 완료
- 무인 장면에서 가장 안정적
- 현재 노트북 테스트에서 INT8보다 빠름

### Raspberry Pi benchmark candidates

1. INT8 MinMax
2. INT8 Percentile

INT8은 현재 최종 모델이 아니라 Raspberry Pi에서 속도 이득과 False Positive trade-off를 확인하기 위한 후보입니다.

## 현재 프로젝트 단계

```text
Training
→ FP32 ONNX
→ INT8 Quantization
→ Laptop Camera Validation  ✓
→ Raspberry Pi Real-Camera Benchmark  ← 다음
→ Final Deployment Selection
```

최종 목표는 PC에서 가장 빠른 모델이 아니라,
**Raspberry Pi 4에서 사람을 안정적으로 검출하면서 RC카의 다른 프로세스를 방해하지 않는 모델**입니다.
