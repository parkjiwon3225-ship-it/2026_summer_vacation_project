# INT8 경량화 모델 발전 과정과 최종 후보 비교

작성일: 2026-08-13

이 폴더는 지금까지 만든 **INT8 경량화 모델만** 시간 순서대로 모아 비교하는 모델 카탈로그입니다. 초기 변환 연습부터 현재 자체 모델 기반 후보까지, 모델을 만든 목적·바꾼 조건·얻은 결과·채택 또는 제외 이유를 함께 기록합니다.

## 결론부터

- `2026-08-07` 모델은 **선생님 제공 결과 코드 기반의 초기 임시 INT8**입니다. 경량화 절차를 익히고 ONNX Runtime 실행을 확인한 개발 이력이지, 현재 프로젝트의 최종 후보가 아닙니다.
- `2026-08-12` 모델은 최종 모델 완성 전에 캘리브레이션 팀과 Raspberry Pi 연결을 먼저 검증한 **통합 시험용 임시 INT8**입니다.
- `2026-08-13` 두 모델은 우리가 직접 설계·학습한 `results.4` FPN48에서 만든 **현재 최종 INT8 후보**입니다.
- 현재 INT8 정확도 1순위는 `QDQ Percentile`입니다. 같은 FP32 기준 모델보다 mAP50:95는 약 `13.9%`, F1은 약 `7.6%` 낮아졌지만 MinMax보다 정확도 보존이 좋았습니다.
- 최종 배포본 확정은 아직 아닙니다. 최신 두 INT8 후보와 FP32 기준 모델을 Raspberry Pi에서 같은 조건으로 비교한 뒤, 정확도 손실을 보상할 만큼 속도·발열 이득이 있는지 확인해야 합니다.

> 초기 계열과 최신 계열은 구조·학습 데이터·평가 절차가 다릅니다. 따라서 파일 크기와 속도는 참고할 수 있지만, 기록되지 않은 초기 mAP를 추정하거나 서로의 정확도를 직접 비교하지 않습니다.

## 모아 둔 INT8 모델

| 단계 | 모델 파일 | 역할 |
|---|---|---|
| 초기 임시본 | [`2026-08-07_teacher-code-v2_int8.onnx`](models/2026-08-07_teacher-code-v2_int8.onnx) | 선생님 제공 결과 코드 기반 변환·실행 확인 |
| 통합 검증본 | [`2026-08-12_preliminary-integration_int8.onnx`](models/2026-08-12_preliminary-integration_int8.onnx) | 캘리브레이션 연동과 Raspberry Pi 카메라 시험 |
| 현재 후보 A | [`2026-08-13_results4-fpn48_qdq-percentile_int8.onnx`](models/2026-08-13_results4-fpn48_qdq-percentile_int8.onnx) | **INT8 정확도 우선 후보** |
| 현재 후보 B | [`2026-08-13_results4-fpn48_qdq-minmax_int8.onnx`](models/2026-08-13_results4-fpn48_qdq-minmax_int8.onnx) | calibration 방식 비교용 후보 |

이 폴더의 모델은 날짜별 원본을 찾기 쉽게 한곳에 복사한 것입니다. [`MODEL_SHA256.txt`](MODEL_SHA256.txt)의 해시가 원본과 같으므로 가중치나 그래프가 수정된 별도 모델은 아닙니다.

## 1. INT8 모델 사양 비교

| 날짜 | 기반 모델과 구조 | 파라미터 | 입력 → 출력 | 양자화와 calibration | INT8 크기 | 계열 FP32 대비 | 상태 |
|---|---|---:|---|---|---:|---:|---|
| 08-07 | 선생님 제공 v2, DSConv backbone + multi-anchor head | 278,361 | `[1,3,240,320]` → `[1,5,5,30,40]` | Static INT8, QInt8 activation/weight, 최대 100장, 직접 resize | 0.301 MiB | 71.76% 감소 | Legacy |
| 08-12 | 임시 v4, DSConv + Residual 단일 15×20 head | 2,868,677 | `[1,3,240,320]` → `[1,5,15,20]` | QDQ, QUInt8/QInt8, per-channel MinMax, 카메라 26장, 직접 resize | 2.817 MiB | 74.26% 감소 | Legacy/통합 검증 |
| 08-13 | 자체 `results.4`, DSConv + Residual + P2~P5 FPN48 + anchor-free head | 323,546 | `[1,3,240,320]` → boxes `[1,6380,4]`, scores `[1,6380]` | QDQ, QUInt8/QInt8, per-channel Percentile, grouped Train 192장, letterbox | 1.320 MiB | 36.23% 감소 | **현재 후보 A** |
| 08-13 | 위와 같은 `results.4` | 323,546 | 위와 같음 | QDQ, QUInt8/QInt8, per-channel MinMax, grouped Train 192장, letterbox | 1.320 MiB | 36.23% 감소 | 현재 후보 B |

FP32 크기는 외부 데이터 파일까지 합산했습니다. 08-07 FP32는 `.onnx + .onnx.data = 1.066 MiB`, 08-12 FP32는 `10.942 MiB`, 08-13 FP32는 `.onnx + .onnx.data = 2.070 MiB`입니다.

## 2. 확인된 성능 비교

### 정답 라벨 기반 정확도

| INT8 모델 | 평가 데이터 | mAP50:95 | Precision | Recall | F1 | Tiny Recall | Small Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| 08-07 초기 임시본 | 정답 기반 기록 없음 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| 08-12 통합 검증본 | 정답 기반 기록 없음 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| 08-13 QDQ Percentile | grouped Valid 1,531장 | **0.217279** | **0.5488** | **0.5084** | **0.5278** | **0.0541** | **0.4600** |
| 08-13 QDQ MinMax | grouped Valid 1,531장 | 0.206241 | 0.5158 | 0.5052 | 0.5105 | 0.0518 | 0.4450 |

최신 계열의 FP32 기준은 mAP50:95 `0.252454`, Precision `0.5863`, Recall `0.5572`, F1 `0.5714`, Tiny Recall `0.0725`, Small Recall `0.5294`입니다. 이 기준과 비교하면:

| 변화 | QDQ Percentile | QDQ MinMax |
|---|---:|---:|
| mAP50:95 | 13.93% 하락 | 18.31% 하락 |
| F1 | 7.62% 하락 | 10.66% 하락 |
| Small Recall | 13.12% 하락 | 15.94% 하락 |

### 속도·현장 검증

| INT8 모델 | 확인 환경 | 평균 추론 | P95 추론 | 추가 결과 |
|---|---|---:|---:|---|
| 08-07 초기 임시본 | ONNX Runtime | 수치 기록 없음 | 수치 기록 없음 | 입력·출력 및 실행 PASS |
| 08-12 통합 검증본 | 노트북 CPU | 2.60 ms | 2.87 ms | FP32 대비 검출 수 일치 92%, 평균 bbox IoU 0.949 |
| 08-12 통합 검증본 | Raspberry Pi 4B, OV5647, 2 threads | 90.63 ms | 95.65 ms | sensor-to-result 137.75 ms, 최고 72.55°C, throttling 없음 |
| 08-13 QDQ Percentile | Windows CPU, ONNX Runtime 2 threads | 16.02 ms | 19.66 ms | Raspberry Pi 실측 대기 |
| 08-13 QDQ MinMax | Windows CPU, ONNX Runtime 2 threads | 18.57 ms | 25.92 ms | Raspberry Pi 실측 대기 |

08-12와 08-13의 노트북 수치는 서로 다른 모델·그래프·시험 환경에서 측정했으므로 직접적인 속도 순위로 사용하지 않습니다. 특히 최신 FPN48에서는 같은 Windows 환경의 FP32가 평균 `8.75 ms`로 INT8보다 빨랐습니다. INT8이라고 항상 빨라지는 것이 아니므로 Raspberry Pi 실측이 최종 판단 기준입니다.

## 3. 왜 이렇게 변화시켰는가

### 1단계 — 경량화 도구 흐름 확인

선생님 제공 결과 코드의 v2 체크포인트를 ONNX로 내보내고 Static INT8로 변환했습니다. 목표는 작은 파일을 만드는 것보다 다음 전체 흐름이 작동하는지 확인하는 것이었습니다.

```text
PyTorch checkpoint → ONNX FP32 → calibration → ONNX INT8 → ONNX Runtime 실행
```

실행은 확인했지만 group-aware 데이터 분할, 정답 기반 경량화 정확도, Raspberry Pi 실제 지연시간 기록이 없으므로 최종 모델 후보로 삼지 않았습니다.

### 2단계 — 다른 팀과 실제 장치 연결을 먼저 확인

최종 모델을 기다리는 동안 임시 v4 모델을 FP32/INT8로 만들었습니다. calibration을 카메라 이미지 26장으로 수행하고, 같은 100프레임에서 FP32와 INT8의 검출 수·confidence·bbox IoU를 비교했습니다. 이어서 Raspberry Pi 4B와 OV5647 실시간 영상에서 다음을 확인했습니다.

- ONNX Runtime이 Raspberry Pi에서 실행되는가
- 카메라 색상과 bbox 좌표가 맞는가
- 2/3/4 threads의 속도와 발열 차이는 어떤가
- 추론 시간뿐 아니라 sensor-to-result 지연시간을 기록할 수 있는가

이 단계에서 2 threads는 평균 추론 약 `90.6 ms`로 3~4 threads보다 느렸지만, 냉각 장치가 없는 조건에서 75°C 열 중단 없이 안정적이었습니다. 이 결과는 최신 모델의 정확도가 아니라 **배포 파이프라인과 장치 운용 조건**을 정한 자료입니다.

### 3단계 — 우리 프로젝트용 모델을 직접 설계하고 평가

최종 후보는 선생님 제공 v2가 아니라 직접 만든 `results.4` FPN48입니다.

```text
DSConv + Residual 경량 backbone
  → 작은 사람을 위한 P2 포함 P2/P3/P4/P5 FPN
  → anchor-free classification + quality + LTRB regression head
```

데이터는 source group 단위로 다시 나눠 leakage를 `0`으로 만들었고, calibration은 grouped Train에서 균등 간격으로 고른 192장에만 적용했습니다. 양자화 품질은 calibration에 사용하지 않은 grouped Valid 전체 1,531장에서 mAP·P·R·F1·크기별 Recall로 평가했습니다.

### 4단계 — 양자화 방식 자체를 비교

- `QDQ Percentile`: activation의 극단적인 일부 값을 제외해 대부분의 값에 8-bit 구간을 더 촘촘하게 배분합니다. MinMax보다 mAP, F1, 작은 사람 Recall이 모두 높았습니다.
- `QDQ MinMax`: 관측된 최솟값부터 최댓값까지 전부 8-bit 범위에 넣습니다. 이상치까지 포함되면서 일반 구간의 양자화 해상도가 낮아졌고 정확도 손실이 더 컸습니다.
- `QOperator MinMax`: 모델 생성과 실행에는 성공했지만 grouped Valid에서 오검출이 크게 늘었습니다. 실험 증거는 보존하되 현장 전달 모델에서는 제외했습니다.

## 4. 최신 후보를 초기 임시본보다 우선하는 이유

초기 임시본의 파일이 더 작다는 사실만으로 더 좋은 모델이라고 판단할 수 없습니다. 최신 후보를 우선하는 이유는 다음과 같습니다.

1. 모델 구조를 RC카의 `320×240` 영상과 작은 사람 검출에 맞게 직접 설계했습니다.
2. 원본·영상 그룹 누수를 제거한 데이터 분할로 일반화 성능을 평가합니다.
3. 학습 전처리와 INT8 calibration 전처리를 같은 letterbox 방식으로 맞췄습니다.
4. runtime PASS가 아니라 grouped Valid 전체의 mAP·F1·크기별 Recall까지 확인했습니다.
5. FP32와 INT8을 같은 기반 가중치에서 만들어 양자화로 인한 손실만 분리해 비교할 수 있습니다.
6. Raspberry Pi의 속도·발열·전체 지연시간까지 최종 선택 기준에 포함합니다.

즉 최신 모델은 초기 모델보다 무조건 정확하다고 수치를 만들어 주장하는 것이 아니라, **우리 프로젝트에 맞는 구조와 신뢰할 수 있는 평가 절차를 갖춘 유일한 최종 후보 계열**이라는 점이 핵심입니다.

## 5. 현재 선택과 다음 갱신

현재 선택 상태는 다음과 같습니다.

```text
INT8 정확도 우선 후보 : QDQ Percentile
INT8 비교/대조 후보   : QDQ MinMax
제외                  : QOperator MinMax
최종 배포본           : Raspberry Pi 3종 비교 후 결정
```

Raspberry Pi 시험에서는 FP32, Percentile, MinMax를 같은 카메라·장면·시작 온도·2 threads로 비교합니다. 다음 항목을 이 문서에 추가하면 최종 결정을 내릴 수 있습니다.

- 평균/P95 inference와 sensor-to-result
- FPS와 최대 온도, throttling 여부
- 빈 배경 오검출과 중복 박스
- 가까운 부분 인물, 먼 작은 사람, 이동하는 사람의 시각 확인
- 정확도 손실 대비 실제 속도·열 개선 폭

최종 선택 후에는 이 문서의 상태를 `후보`에서 `최종 배포본`으로 바꾸고, 선택 모델의 Raspberry Pi 수치와 선택 이유를 고정 기록합니다. 최종 Test split은 후보 선택이 끝난 뒤 한 번만 사용합니다.

## 6. 원본 자료 위치

- 초기 선생님 코드 기반 변환: [`../2026-08-07_legacy-int8-onnx`](../2026-08-07_legacy-int8-onnx/)
- 임시 캘리브레이션 전달본: [`../2026-08-12_preliminary-calibration-handoff`](../2026-08-12_preliminary-calibration-handoff/)
- 임시 모델 Raspberry Pi 실측: [`../2026-08-12_raspberry-pi-first-benchmark`](../2026-08-12_raspberry-pi-first-benchmark/)
- 최신 `results.4` FP32/INT8 생성·평가·Pi 시험 코드: [`../2026-08-13_results4-pi-model-variants`](../2026-08-13_results4-pi-model-variants/)
- 전체 수치 CSV: [`MODEL_COMPARISON.csv`](MODEL_COMPARISON.csv)

