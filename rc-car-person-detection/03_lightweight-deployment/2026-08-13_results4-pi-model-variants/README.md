# Results.4 FPN48 Raspberry Pi 3모델 비교 테스트

작성일: 2026-08-13

이 묶음은 1차 병렬 학습에서 가장 좋았던 `results.4`의 epoch 37 `best.pt`를 기준으로 만든 Raspberry Pi 현장 비교용 모델입니다. 모델 구조와 학습 가중치는 같고 ONNX 배포 정밀도와 INT8 calibration 방식만 다릅니다.

## 결론부터

내일 세 모델을 같은 장소·사람 거리·조명·실행 시간으로 비교합니다.

1. `FP32`: 정확도 기준 모델이며 **현재 1순위 출발점**
2. `INT8 QDQ Percentile`: INT8 중 정확도 우선 후보
3. `INT8 QDQ MinMax`: INT8 비교 후보

`INT8 QOperator MinMax`도 만들고 전체 Valid에서 평가했지만 오검출이 크게 증가해 현장 전달 대상에서 제외했습니다.

이번 FPN48 모델은 이미 파라미터가 323,546개로 작습니다. INT8은 FP32 전체 파일 크기보다 약 36% 작아졌지만, Windows CPU 시험에서는 오히려 느렸습니다. Raspberry Pi에서도 반드시 INT8이 더 빠르다고 가정하지 말고 실제 결과를 기준으로 선택해야 합니다.

## 모델 목록

| 모델 | 형식 | 크기 | grouped Valid mAP50:95 | P | R | F1 | small R |
|---|---|---:|---:|---:|---:|---:|---:|
| `results4_fpn48_fp32.onnx` + `.onnx.data` | FP32 | 2.070 MiB | **0.252454** | **0.5863** | **0.5572** | **0.5714** | **0.5294** |
| `results4_fpn48_int8_qdq_percentile.onnx` | Static INT8 QDQ | 1.320 MiB | 0.217279 | 0.5488 | 0.5084 | 0.5278 | 0.4600 |
| `results4_fpn48_int8_qdq_minmax.onnx` | Static INT8 QDQ | 1.320 MiB | 0.206241 | 0.5158 | 0.5052 | 0.5105 | 0.4450 |

FP32 ONNX는 PyTorch 원본 mAP `0.252478`을 사실상 그대로 재현했습니다. INT8 Percentile의 mAP 하락은 FP32 대비 약 13.9%, MinMax는 약 18.3%입니다.

### 차이 해석

- **FP32**: 학습 가중치를 32-bit 부동소수점으로 유지합니다. 정확도 기준선이며 양자화 오차가 없습니다.
- **INT8 QDQ Percentile**: Train 이미지의 activation 분포에서 극단적인 일부 값을 제외한 범위를 기준으로 8-bit 양자화합니다. MinMax보다 작은 사람과 전체 mAP 보존이 좋았습니다.
- **INT8 QDQ MinMax**: 관찰된 최솟값과 최댓값 전체를 8-bit 범위에 넣습니다. 이상치까지 포함하면서 유효 해상도가 줄어 Percentile보다 정확도 하락이 컸습니다.
- **QDQ**: 그래프에 Quantize/Dequantize 연산을 넣는 ONNX Runtime 호환 방식입니다. 실제 연산 속도는 Pi의 CPU와 Runtime 최적화에 따라 달라집니다.

| 비교 항목 | Percentile vs FP32 | MinMax vs FP32 |
|---|---:|---:|
| 파일 크기 | 약 36.2% 감소 | 약 36.2% 감소 |
| mAP50:95 | 약 13.9% 하락 | 약 18.3% 하락 |
| F1 | 약 7.6% 하락 | 약 10.7% 하락 |
| Small recall | 약 13.1% 하락 | 약 16.0% 하락 |

따라서 경량화 여부는 파일 크기만으로 결정하지 않습니다. 실제 Raspberry Pi의 지연시간·발열 개선이 위 정확도 손실을 보상하는지를 확인해야 합니다.

## 중요한 평가 조건

- 입력: OV5647 `640×480`
- 모델 입력: RGB `320×240` letterbox, padding 114
- score threshold: `0.25`
- NMS IoU threshold: `0.50`
- ONNX Runtime 내부 threads: `2`
- 온도 중단: `75°C`
- 프레임·영상 저장 없음; 수치 CSV/JSON만 저장

모델마다 threshold를 바꾸면 공정 비교가 아니므로 첫 비교에서는 위 값을 유지합니다.

다음 조건도 동일하게 유지해야 합니다.

- 같은 Raspberry Pi와 전원 상태
- 같은 냉각 조건과 비슷한 시작 온도
- 같은 카메라 위치와 고정된 pan/tilt 각도
- 같은 사람, 거리, 이동 방향과 장면별 지속 시간
- 다른 프로그램의 CPU 사용을 최소화
- 모델별 시험 전에 warmup 완료

시험 순서를 항상 FP32부터 고정하면 뒤 모델이 더 뜨거운 상태에서 실행되는 순서 편향이 생길 수 있습니다. 1차 확인은 FP32 → Percentile → MinMax로 하되, 최종 두 후보는 충분히 냉각한 뒤 순서를 바꿔 한 번 더 측정하는 것이 좋습니다.

## Raspberry Pi로 옮긴 뒤 순서

```bash
cd ~/Desktop/RC_RESULTS4_PI_VARIANTS
chmod +x 01_setup_pi.sh 02_test_one_model.sh 03_run_three_models.sh
./01_setup_pi.sh
.venv/bin/python 04_check_models.py
```

`04_check_models.py`에서 세 파일 모두 `PASS`가 나와야 합니다.

## 권장 테스트 순서

열 영향을 줄이기 위해 FP32부터 하나씩 시험합니다. 각 모델 사이에는 온도가 충분히 내려갈 시간을 둡니다.

```bash
./02_test_one_model.sh models/results4_fpn48_fp32.onnx 180
./02_test_one_model.sh models/results4_fpn48_int8_qdq_percentile.onnx 180
./02_test_one_model.sh models/results4_fpn48_int8_qdq_minmax.onnx 180
```

세 모델을 자동 순서로 실행할 수도 있습니다.

```bash
./03_run_three_models.sh 120
```

다만 자동 실행은 모델 사이 냉각 시간이 없으므로 정확한 온도 비교에는 개별 실행을 권장합니다.

## 장면 조건

각 모델에 동일한 순서와 시간을 적용합니다.

1. 빈 배경 20초
2. 가까운 사람 30초
3. 중간 거리 사람 30초
4. 먼 사람 30초
5. 좌우로 움직이는 사람 30초
6. 화면 경계 또는 부분 인물 30초

실행 중 숫자 키로 장면을 표시할 수 있습니다.

- `0`: empty
- `1`: near
- `2`: middle
- `3`: far
- `4`: moving
- `5`: multiple_people
- `q`: 종료

현재 코드에는 부분 인물 전용 키가 없으므로 부분 인물 시험은 `near`로 기록하고 별도 메모에 남깁니다.

## 화면에서 볼 것

- 사람에게 박스가 맞게 생기는가
- 한 사람에게 중복 박스가 생기는가
- 빈 배경에서 잘못 검출하는가
- 먼 작은 사람과 움직이는 사람을 놓치는가
- 상체·하체·화면 경계의 부분 인물을 검출하는가
- 색상이 정상인가
- 박스가 원본 640×480 프레임 위치와 맞는가

## 수치로 비교할 것

각 실행의 `summary.json`에서 다음 값을 비교합니다.

- `inference_ms.mean`, `inference_ms.p95`
- `sensor_to_result_ms.mean`, `sensor_to_result_ms.p95`
- `temperature_c.max`
- `throttled_end`
- `frames_with_people_percent`
- `detections_per_frame`
- confidence mean/median/P95

실제 정답 라벨이 없는 카메라 시험의 검출 수와 confidence는 정확도 자체가 아닙니다. 속도·발열·명백한 오검출·박스 위치를 확인하는 현장 시험입니다. 정확도 비교는 이미 grouped Valid에서 수행했습니다.

## 결과 위치와 회수

```text
pi_variant_results/
├─ results4_fpn48_fp32/<실행시각>/
├─ results4_fpn48_int8_qdq_percentile/<실행시각>/
└─ results4_fpn48_int8_qdq_minmax/<실행시각>/
```

시험 후 `pi_variant_results` 폴더 전체를 USB에 복사해 AI 팀에 전달합니다.

## 모델 선택 기준

1. FP32가 8 FPS 이상, 평균 sensor-to-result 200ms 이하이고 75°C 미만이면 FP32를 우선합니다.
2. INT8이 실제 Pi에서 유의하게 빠르고 박스·오검출 문제가 없을 때만 INT8을 선택합니다.
3. INT8 두 모델의 속도가 비슷하면 Valid 정확도가 높은 Percentile을 선택합니다.
4. 어떤 모델도 자동 좌표 계산에 바로 확정하지 않습니다. bbox가 프레임 경계에 닿는 부분 인물은 발 위치가 유효하지 않을 수 있습니다.

### 현장 합격 기준

| 항목 | 목표 | 주의 기준 |
|---|---:|---:|
| 결과 FPS | 8 FPS 이상 | 5 FPS 미만 |
| 평균 sensor-to-result | 200 ms 이하 | 300 ms 초과 |
| P95 sensor-to-result | 250 ms 이하 | 500 ms 초과 |
| 최대 온도 | 75°C 미만 | 75°C 도달로 자동 중단 |
| Throttling | 없음 | 발생 시 부적합 |
| 빈 배경 | 반복 오검출 없음 | 지속·중복 검출 발생 |

이 기준은 AI가 모터를 직접 제어하지 않고 지도 표시 정보를 제공하는 현재 프로젝트 목적에 맞춘 1차 목표입니다.

## 캘리브레이션 팀이 특히 확인할 정보

`detections.csv`에는 모델 입력 좌표와 원본 카메라 좌표가 함께 기록됩니다.

- `model_xmin`~`model_ymax`: 320×240 letterbox 좌표
- `camera_xmin`~`camera_ymax`: padding과 scale을 제거한 원본 640×480 좌표
- `bottom_center_x`, `bottom_center_y`: 원본 프레임 기준 bbox 하단 중심점
- `normalized_bottom_center_x/y`: 원본 프레임 기준 0~1 정규화 좌표
- `timestamp`, `frame`: 검출 결과와 센서 정보를 대응시키는 식별값

중요한 제한:

- bbox가 화면 아래 경계에 닿으면 하단 중심점이 실제 발 위치가 아닐 수 있습니다.
- 상체·하체만 보이는 부분 인물은 사람 존재 확인에는 사용할 수 있지만 정확한 지면 위치 계산에는 바로 사용하면 안 됩니다.
- 실제 거리·지도 좌표에는 카메라 내부 행렬, 왜곡 계수, 설치 자세, GPS·IMU·pan/tilt의 timestamp 정합이 추가로 필요합니다.
- 이 카메라 시험에는 정답 라벨이 없으므로 검출 수가 많다고 정확한 모델이라는 뜻은 아닙니다.

## 생성 및 검증 기록

- calibration: grouped Train 192장, 균등 간격 deterministic sample
- 검증: grouped Valid 전체 1,531장
- ONNX checker: 세 모델 PASS
- ONNX Runtime CPU dummy inference: 세 모델 PASS
- 출력 shape: boxes `[1, 6380, 4]`, scores `[1, 6380]`
- 상세 결과: `build_report.json`, `valid_metrics.json`, `MODEL_SHA256.txt`

FP32의 `results4_fpn48_fp32.onnx.data`는 외부 weight 파일이므로 `.onnx`와 항상 같은 폴더에 함께 두어야 합니다.
