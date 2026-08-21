# RC-Car 사람 감지 AI — 개발·경량화 기록

Raspberry Pi 기반 RC카가 낮 시간 운동장에서 여러 사람을 실시간 감지하도록 만든 AI 파트의 전체 기록입니다. 발표 자료가 아니라 개발자가 설계와 실험을 재현하고, 선택 근거를 검토하고, 최종 모델을 통합할 수 있도록 구성했습니다.

> **최종 배포 모델:** `TAIL004` (`person_detector_tail004_int8.onnx`)
>
> **학습 원본:** `R46 / r46_final448_seed15_100e / 448×336 / best epoch 25`
>
> **배포 방식:** OpenCV C++ 카메라 → ONNX Runtime C++ CPU → 선택적 QDQ INT8
>
> **최종 카메라 입력:** 320×240, BGR, 고정 화각
>
> **마지막 정리:** 2026-08-21 (2026-08-20까지의 개발·선택 결과 반영)

## 빠른 길찾기

| 목적 | 위치 |
|---|---|
| 최종 모델 다운로드·사용 | [`05_final-release/2026-08-20_tail004`](05_final-release/2026-08-20_tail004/) |
| 최종 모델 카드 | [`MODEL_CARD.md`](05_final-release/2026-08-20_tail004/MODEL_CARD.md) |
| C++ 통합 규격 | [`INTEGRATION_CONTRACT.md`](05_final-release/2026-08-20_tail004/INTEGRATION_CONTRACT.md) |
| 최종 학습 후보 비교 | [`2026-08-20_final-resolution-search`](02_training-experiments/2026-08-20_final-resolution-search/) |
| TAIL004 경량화 비교 | [`2026-08-20_r46-tail004-final`](03_lightweight-deployment/2026-08-20_r46-tail004-final/) |
| Hard-negative 실험 | [`2026-08-19_hard-negative-study`](02_training-experiments/2026-08-19_hard-negative-study/) |
| 초기부터의 모델 개발 | [`01_model-development`](01_model-development/) |

## 1. 최종 모델 요약

### 학습 모델 R46

| 항목 | 값 |
|---|---:|
| 입력 크기 | 448×336 |
| best epoch | 25 |
| validation mAP50:95 | 0.261832 |
| AP50 | 0.561210 |
| AP75 | 0.210810 |
| Precision | 0.817398 |
| Recall | 0.412851 |
| F1 | 0.548610 |
| tiny `<16 px` recall | 0.002104 |
| small `16–32 px` recall | 0.155005 |

### 최종 경량화 TAIL004

| 항목 | 값 |
|---|---:|
| ONNX 파일 | `person_detector_tail004_int8.onnx` |
| 크기 | 734,360 bytes (약 0.700 MiB) |
| FP32 대비 크기 감소 | 47.25% |
| 양자화된 Conv | 19개 |
| confidence 0.25 detection 유지율 | 97.90% |
| FP32 box 평균 IoU | 0.9746 |
| SHA-256 | `230755c15376065bdfbcea44cbc9259d9691ec48319bc96bdbfb15c38b3e01be` |

TAIL004는 모든 계층을 INT8로 바꾸지 않습니다. 민감한 early backbone, FPN, detection head를 FP32로 유지하고 상대적으로 안전한 중·후반 Conv를 QDQ INT8로 바꾼 mixed-precision 모델입니다.

## 2. 모델 구조

YOLO 패키지를 선택한 것이 아니라 프로젝트 목적에 맞춰 처음부터 만든 custom anchor-free detector입니다.

```text
RGB image
  → DSConv + Residual lightweight backbone
  → P2–P5 lightweight FPN (final source: 48 channels)
  → anchor-free detection head
  → classification + quality + LTRB box regression
  → decode + NMS
```

- `DSConv`: 일반 convolution을 depthwise와 pointwise로 분리해 연산량을 절감
- `Residual`: 경량 네트워크에서도 학습 안정성과 정보 흐름을 보완
- `P2–P5 FPN`: 서로 다른 크기의 사람, 특히 작은 사람을 여러 해상도 feature에서 탐지
- `Anchor-free`: 미리 정한 anchor box 대신 위치별 class·quality·box 거리를 직접 예측
- 단일 class `person`: RC카의 사람 감지 목적에 계산을 집중

## 3. 날짜별 개발 과정과 판단

| 날짜 | 단계 | 비교·판단 |
|---|---|---|
| 08-06 | 데이터 준비 | 원본 split의 난이도 편향과 동일 원본 leakage를 확인. 원본·영상 그룹이 split을 넘지 않는 group-aware v1 데이터셋으로 재구성 |
| 08-10~12 | 모델 기반 구현 | DSConv+Residual, FPN, anchor-free head, target assignment, loss, metric, checkpoint/resume 파이프라인을 순차 검증 |
| 08-12 | 초기 학습·Pi 시험 | mini-overfit과 GPU 학습 성공. FP32/INT8 ONNX 및 OV5647 카메라 시험으로 배포 경로 검증 |
| 08-13 | 6대 병렬 실험 | learning rate, FPN 폭, loss weight, assignment radius를 비교해 FPN48 계열을 우선 후보로 축소 |
| 08-14 | Round 2·3 | seed와 구조 변수를 분리해 비교. 짧은 screening과 장기 학습을 구분하고 자동 resume·로그 수집을 강화 |
| 08-14~18 | 장기·고해상도 탐색 | 320×240부터 768×576까지 비교. 640×480 계열이 높은 mAP를 보였지만 Pi 배포 계산량과 Recall 균형을 함께 평가 |
| 08-18~20 | 최종 해상도·seed 비교 | 576×432 R33이 mAP 0.3014로 최고였지만, 448×336 R46이 더 작은 입력에서 Recall 0.4129와 small recall 0.1550을 확보해 최종 배포 원본으로 선정 |
| 08-19~20 | Hard-negative | 운동장 오탐 후보 343장을 사용. 일부 모델의 FP는 줄었으나 R46에는 일관된 개선이 없어 원본 R46 checkpoint를 유지 |
| 08-20 | 최종 경량화 | Q3+Q4 control, TAIL004, TAIL006을 비교. 실제 Pi 카메라 결과와 보수적 정확도·크기 균형을 고려해 TAIL004 확정 |

## 4. 주요 선택지와 최종 결정

### Kaggle 원본 split vs group-aware split

원본 split에는 동일 원본의 변형 이미지나 같은 영상 계열이 train/valid/test에 나뉠 가능성이 있었고, 작은 사람 비율도 split마다 달랐습니다. 모든 데이터를 원본·sequence 그룹 단위로 다시 배치해 group leakage `0`인 v1 데이터셋을 사용했습니다.

### 최고 mAP 모델 vs 실제 배포 원본

R33(576×432)은 validation mAP50:95 `0.301354`로 가장 높았습니다. 하지만 R46(448×336)은 입력 픽셀이 약 40% 적고, Recall `0.412851`, small recall `0.155005`로 배포 목적에 더 균형적이었습니다. RC카가 놓치는 사람에 더 민감하고 Raspberry Pi 실시간성이 필요하므로 R46을 선택했습니다.

### FP32 vs full INT8 vs selective INT8

- FP32: 출력 기준이지만 모델 크기와 CPU 비용이 큼
- full INT8: 가장 공격적으로 줄일 수 있으나 이 모델에서는 detection 출력 손상이 큼
- selective INT8: 민감한 부분을 FP32로 남겨 출력 보존성과 크기 절감을 절충

최종적으로 selective QDQ INT8을 채택했습니다.

### TAIL004 vs TAIL006

TAIL006은 더 작고 detection 유지율도 약간 높았지만 box IoU가 더 낮고 추가 detection이 늘었습니다. TAIL004는 FP32 대비 47.25% 감소하면서 평균 box IoU 0.9746을 유지했고 실제 카메라 검증에서도 안정적이어서 최종본으로 선택했습니다. TAIL006은 연구·비교 후보로 보존합니다.

## 5. 최종 실행 흐름

```text
OpenCV C++ camera.read(frame)
  → 320×240 CV_8UC3 BGR 확인
  → capture frame_id + steady_clock timestamp
  → BGR → RGB
  → 448×336 letterbox + /255 + NCHW float32
  → ONNX Runtime C++ CPU inference (TAIL004)
  → score threshold 0.25
  → NMS IoU 0.50
  → box를 원본 320×240 좌표로 복원
  → persons[{x1,y1,x2,y2,confidence}]
```

카메라는 OpenCV가 열고, ONNX Runtime은 전달받은 프레임의 AI 추론만 담당합니다. 30 FPS 카메라 프레임을 모두 큐에 쌓지 않고 최신 프레임 한 장만 AI에 전달하는 구조를 권장합니다.

## 6. 알려진 한계

- `<16 px` 초소형 사람 recall은 매우 낮아 70 m 운동장 전체를 동일한 신뢰도로 탐지하는 모델은 아닙니다.
- 거리(m)는 학습 목표가 아니라 이미지 속 사람 픽셀 크기와 화각에 의해 결정됩니다.
- confidence `0.25`는 최종 시작값이며, 낮 운동장 실측에서 FP/FN을 기록해 조정해야 합니다.
- 새 운동장 구조물을 반복 오탐하면 사람이 없는 이미지를 hard-negative로 추가할 수 있지만, 기존 validation/test와 분리하고 재학습 전후 Recall을 함께 확인해야 합니다.
- 모델의 최종 정확도와 실제 시스템 FPS·온도는 Raspberry Pi의 C++ 통합 실행에서 다시 측정해야 합니다.

## 7. 저장소 구조

```text
ai-model/
├─ 01_model-development/       데이터 분할, 모델·학습 기반 코드
├─ 02_training-experiments/    날짜별 학습 설정, 결과, 비교와 선택
├─ 03_lightweight-deployment/  ONNX/INT8 변환과 Pi·카메라 검증
├─ 04_team-integration/        다른 파트와의 인터페이스 자료
└─ 05_final-release/           최종 TAIL004 배포 패키지
```

## 8. 저장 원칙

1. 각 실험은 날짜별 폴더에서 목적, 설정, 결과, 결론을 함께 기록합니다.
2. 학습은 대표 `best.pt`, config, history, 집계 CSV를 보존하고 중복 `last.pt`는 제외합니다.
3. 배포 모델은 quantization 방식과 SHA-256을 함께 기록합니다.
4. 실패하거나 선택되지 않은 결과도 의사결정에 필요하면 수치와 이유를 남깁니다.
5. 원본 데이터셋, calibration 원본 이미지, 로컬 가상환경, cache, 중복 ZIP, 발표 준비 문서는 GitHub에 올리지 않습니다.
6. 최종 test split은 후보 탐색에 사용하지 않고 최종 확정 모델의 마지막 평가에만 사용합니다.
