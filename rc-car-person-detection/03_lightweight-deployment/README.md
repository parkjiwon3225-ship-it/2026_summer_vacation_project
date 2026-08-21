# 03. 경량화 및 Raspberry Pi 배포

PyTorch checkpoint를 ONNX로 변환하고 FP32, full INT8, selective INT8을 비교한 뒤 실제 카메라와 Raspberry Pi에서 정확도·지연시간·FPS·발열·False Positive를 검증한 기록입니다.

## 날짜별 기록

| 날짜 | 폴더 | 상태 | 설명 |
|---|---|---|---|
| 2026-08-07 | [`legacy-int8-onnx`](2026-08-07_legacy-int8-onnx/) | Legacy | 선생님 제공 초기 Person Detector v2를 임시 ONNX/INT8로 변환 |
| 2026-08-12 | [`preliminary-calibration-handoff`](2026-08-12_preliminary-calibration-handoff/) | Legacy / 인계 | custom 최종 모델 전 팀 연동 확인용 FP32/INT8 |
| 2026-08-12 | [`raspberry-pi-first-benchmark`](2026-08-12_raspberry-pi-first-benchmark/) | 완료 | OV5647 + Pi 1차 실측, thread 수·지연·발열 확인 |
| 2026-08-13 | [`int8-model-evolution-and-finalists`](2026-08-13_int8-model-evolution-and-finalists/) | 완료 | 초기 경량화 모델의 목적과 변화 과정 비교 |
| 2026-08-13 | [`results4-pi-model-variants`](2026-08-13_results4-pi-model-variants/) | 과거 기준 | results.4 FP32 / INT8 MinMax / Percentile 비교 |
| 2026-08-14 | [`results14-onnx-export`](2026-08-14_results14-onnx-export/) | 보관 | results.14 FP32 / INT8 변환본 |
| 2026-08-16 | [`results20-camera-validation`](2026-08-16_results20-camera-validation/) | 완료 | 카메라 FP32/INT8 비교, 출력 이상 원인 진단, 실제 화면 기록 |
| 2026-08-18 | [`results27-selective-int8`](2026-08-18_results27-selective-int8/) | 후보 연구 | early backbone 민감도와 Q1~Q4 selective quantization 탐색 |
| 2026-08-20 | [`r46-tail004-final`](2026-08-20_r46-tail004-final/) | **최종** | R46 FP32, Q3Q4, TAIL004, TAIL006 비교 후 TAIL004 확정 |

## 경량화 발전 과정

```text
PyTorch best.pt
  → FP32 ONNX: 기준 출력과 이식성 확보
  → full INT8: 크기는 줄지만 detector 출력 손상 확인
  → selective INT8: 민감한 계층을 FP32로 유지
  → Q3+Q4 control
  → TAIL004 / TAIL006 tail 확장 비교
  → TAIL004 최종 선택
```

모델 설계 단계의 DSConv·FPN48 경량화와 배포 단계의 INT8 양자화는 서로 다른 작업입니다. 전자는 학습 가능한 네트워크의 연산량·파라미터를 줄이고, 후자는 학습이 끝난 ONNX의 일부 weight·activation 표현을 FP32에서 INT8로 바꿉니다.

## 최종 경량화 비교

| 모델 | INT8 Conv | 크기 (MiB) | FP32 대비 감소 | conf 0.25 유지율 | 평균 box IoU | missed / extra | 결론 |
|---|---:|---:|---:|---:|---:|---:|---|
| R46 FP32 | 0 | 1.327629 | - | 기준 | 1.000000 | 0 / 0 | 정확도 기준 |
| Q3Q4 control | 15 | 0.728728 | 45.11% | 97.71% | **0.976264** | 12 / 17 | 보수적 후보 |
| **TAIL004** | **19** | **0.700340** | **47.25%** | **97.90%** | **0.974578** | **11 / 17** | **최종 배포** |
| TAIL006 | 21 | 0.692083 | 47.87% | **98.28%** | 0.972780 | 9 / 19 | 더 공격적인 연구 후보 |

세 모델 모두 내부 강한 보존 기준을 통과했습니다. TAIL006이 조금 더 작지만 추가 detection과 box 변화가 늘어났고, TAIL004가 크기·출력 보존·실제 카메라 안정성의 중간점이어서 최종 선택됐습니다.

## 최종 모델 특성

- 모델: [`person_detector_tail004_int8.onnx`](../05_final-release/2026-08-20_tail004/models/person_detector_tail004_int8.onnx)
- 양자화: QDQ, signed INT8 activation/weight, per-channel weight
- calibration: 96 images, percentile 99.99
- validation preservation check: 120 images
- 입력 tensor: `[1, 3, 336, 448]`, RGB, float32, `/255`
- 출력: boxes `[1, 12502, 4]`, scores `[1, 12502]`
- 기본 후처리: confidence 0.25, NMS IoU 0.50

## Raspberry Pi 운영 판단

- 카메라 입력 320×240과 모델 입력 448×336은 충돌하지 않습니다. 4:3 비율을 유지한 letterbox 전처리를 사용합니다.
- OpenCV C++가 320×240 BGR 프레임을 획득하고, ONNX Runtime C++가 AI 추론을 실행합니다.
- 30 FPS 카메라라도 AI 입력은 누적하지 않고 최신 프레임 한 장만 유지합니다.
- 속도 평가는 추론 시간뿐 아니라 camera capture부터 결과 생성까지의 end-to-end latency와 실제 처리 FPS를 함께 기록합니다.
- 발열은 오탐·정확도와 별개의 운영 안정성 기준입니다. throttle 여부와 장시간 온도를 실제 통합 프로그램에서 확인합니다.

최종 C++ 통합 세부 규격은 [`INTEGRATION_CONTRACT.md`](../05_final-release/2026-08-20_tail004/INTEGRATION_CONTRACT.md)를 따릅니다.
