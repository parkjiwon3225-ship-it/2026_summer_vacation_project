# R46 selective INT8와 최종 TAIL004 선택

## 결과

최종 배포 모델은 [`models/R46_448_TAIL004_INT8.onnx`](models/R46_448_TAIL004_INT8.onnx)이다.

```text
source checkpoint : r46_final448_seed15_100e / epoch 25
input             : float32 RGB NCHW [1,3,336,448]
outputs           : boxes [1,12502,4], scores [1,12502]
quantization      : QDQ, QInt8 activation + QInt8 weight, per-channel
calibration       : train 96장, Percentile 99.99
INT8 scope        : backbone Q2 tail 4 Conv + Q3 + Q4
FP32 scope        : Q1, Q2 early, FPN, detection head
```

`TAIL004`는 개발 중 이름 `q2tail4_q3q4`를 최종 릴리스에서 읽기 쉽게 고정한 이름이다.

## 왜 full INT8가 아닌 selective INT8인가

앞선 results20/results27 실험에서 전체 Conv를 한 번에 PTQ하면 score와 box 출력이 크게 흔들렸다. backbone을 구간별로 분리한 sensitivity 분석에서 early Q1은 특히 민감했고, late Q3/Q4는 상대적으로 안정적이었다.

따라서 다음 순서로 범위를 넓혔다.

```text
Q3+Q4 control (15 Conv)
  → Q2 tail 4 + Q3+Q4 (19 Conv, TAIL004)
  → Q2 tail 6 + Q3+Q4 (21 Conv, TAIL006)
```

## 오프라인 비교

FP32 R46을 reference로 120장에 동일 전처리·threshold·NMS를 적용한 출력 보존성 결과다.

| Variant | 크기 | FP32 대비 감소 | INT8 Conv | conf 0.25 보존율 | Mean IoU | Missed | Extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q3Q4 control | 0.7287 MB | 45.11% | 15 | 97.71% | **0.9763** | 12 | 17 |
| **TAIL004** | **0.7003 MB** | **47.25%** | **19** | **97.90%** | 0.9746 | **11** | **17** |
| TAIL006 | 0.6921 MB | 47.87% | 21 | **98.28%** | 0.9728 | 9 | 19 |

세 후보 모두 사전 보존성 gate에서 `PROJECT_STRONG_PASS`였다. TAIL006이 가장 작고 retain 수치도 높았지만 extra detection과 양자화 범위가 늘었다. 최종 선택은 라즈베리파이 실기 관찰까지 포함해 속도·검출 안정성의 중간점인 TAIL004로 고정했다. TAIL006은 연구 대안, Q3Q4는 회귀 확인용 control로 보존한다.

PC의 QDQ latency는 Raspberry Pi 결론으로 사용하지 않는다. 같은 PC에서도 INT8이 FP32보다 느렸으며, 최종 판단은 ARM ONNX Runtime과 전체 시스템 부하에서 해야 한다.

## R46이 가진 정확도

양자화 보존율은 ground-truth mAP가 아니다. TAIL004는 다음 R46 FP32 성능을 최대한 보존하는 모델이다.

| mAP50:95 | AP50 | AP75 | Precision | Recall | F1 | Tiny recall | Small recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.261832 | 0.561210 | 0.210810 | 0.817398 | 0.412851 | 0.548610 | 0.002104 | 0.155005 |

특히 tiny person recall은 낮으므로 먼 사람 검출 한계가 남아 있다. 이 한계를 TAIL004의 오프라인 보존율과 혼동하지 않는다.

## 파일 안내

```text
models/
  R46_448_FP32_REFERENCE.onnx
  R46_448_Q3Q4_INT8_CONTROL.onnx
  R46_448_TAIL004_INT8.onnx       # 최종 선택
  R46_448_TAIL006_INT8.onnx

scripts/                          # export, selective quantization, 검증 코드
reports/                          # node group, raw output·detection agreement
quantization_comparison.csv       # 핵심 비교표
MODEL_SHA256.txt                  # 모델 무결성
```

## 재실행 순서

이 폴더의 스크립트는 원래 self-contained 빌드 패키지에서 실행됐다. 재현 시 `source/R46_original_best.pt`, `src/rc_detector/`, calibration 96장, validation 120장을 준비한 뒤 다음 순서로 실행한다.

```text
01_export_fp32.py
02_quantize_extra_light.py
03_validate_extra_light.py
```

원본 이미지와 중복 build ZIP은 GitHub에 포함하지 않는다. source checkpoint와 training history는 `02_training-experiments/2026-08-20_final-resolution-search/`에 있다.
