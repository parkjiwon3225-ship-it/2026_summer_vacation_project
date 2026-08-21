# Raspberry Pi FP32 / INT8 1차 벤치마크

RC-Car Person Detection 모델을 Raspberry Pi 4B CPU에서 실제 카메라와 함께 실행해 보는 1차 배포 시험 코드입니다.

이 시험의 목적은 최종 정확도를 결정하는 것이 아니라 다음 사항을 먼저 확인하는 것입니다.

- ONNX FP32와 Static INT8 모델이 Raspberry Pi에서 정상적으로 로드되는가
- 카메라 입력을 포함했을 때 실제 처리속도와 지연시간은 어느 정도인가
- INT8 경량화가 속도, 메모리 및 온도에 어떤 차이를 만드는가
- 같은 프레임에서 FP32와 INT8의 검출 개수, confidence, bounding box가 얼마나 일치하는가
- 장시간 CPU 추론 중 저전압 또는 thermal throttling이 발생하는가

> 현재 사용 모델은 최종 학습 모델이 아니라 Raspberry Pi 실행 경로를 검증하기 위한 임시 모델입니다. 최종 `best.pt` 선정 후 같은 시험을 다시 실행해야 합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `pi_first_benchmark.py` | 측정 및 결과 저장 프로그램 |
| `pi_int8_camera_test.py` | VNC에서 INT8 사람 검출을 확인하고 수치를 저장하는 주 시험 프로그램 |
| `01_setup_pi.sh` | Raspberry Pi 최초 환경 설치 |
| `02_run_int8_camera_test.sh` | INT8 카메라 시각 시험 실행 |
| `03_run_fp32_int8_benchmark.sh` | FP32/INT8 비교 벤치마크 실행 |
| `requirements-pi.txt` | Python 패키지 목록 |
| `.gitignore` | 가상환경과 측정 결과 제외 |

이 시험에서 사용한 임시 모델은 같은 상위 파트의 `2026-08-12_preliminary-calibration-handoff`에 있습니다. 현재 배포 후보는 `2026-08-13_results4-pi-model-variants`를 사용하세요.

- `person_detector_fp32.onnx`
- `person_detector_int8.onnx`

실행 전 두 모델을 이 README와 같은 폴더로 복사해야 합니다.

```text
2026-08-12_raspberry-pi-first-benchmark/
├── person_detector_fp32.onnx
├── person_detector_int8.onnx
├── pi_first_benchmark.py
├── pi_int8_camera_test.py
├── requirements-pi.txt
├── 01_setup_pi.sh
├── 02_run_int8_camera_test.sh
└── 03_run_fp32_int8_benchmark.sh
```

## 모델 입출력

입력:

```text
Shape  : [1, 3, 240, 320]
Layout : NCHW
Type   : float32
Color  : RGB
Range  : 0.0 ~ 1.0
```

출력:

```text
Shape : [1, 5, 15, 20]

0 : objectness logit
1 : center X offset logit
2 : center Y offset logit
3 : bounding-box width logit
4 : bounding-box height logit
```

현재 임시 후처리 설정은 confidence `0.20`, NMS IoU `0.20`입니다.

## Raspberry Pi 준비

Raspberry Pi 터미널에서 이 폴더로 이동합니다.

```bash
cd /복사한/경로/2026-08-12_raspberry-pi-first-benchmark
```

최초 한 번만 다음을 실행합니다.

```bash
chmod +x 01_setup_pi.sh 02_run_int8_camera_test.sh 03_run_fp32_int8_benchmark.sh
./01_setup_pi.sh
```

스크립트는 다음 작업을 수행합니다.

1. `python3-venv`, `python3-opencv` 설치
2. CSI 카메라용 Picamera2와 OpenCV 설치
3. 시스템 패키지를 사용할 수 있는 `.venv` 생성
4. ONNX Runtime과 `psutil` 설치
5. 필수 라이브러리 import 검사

마지막에 `SETUP PASS`가 출력되어야 합니다.

카메라 장치를 확인합니다.

```bash
ls -l /dev/video*
```

기본 카메라 번호는 `0`입니다. OV5647 CSI 카메라는 RAW 장치인 `/dev/video0`을 OpenCV로 직접 열지 않고 Picamera2/libcamera를 사용합니다. `rpicam-hello --list-cameras`의 번호가 다르면 실행 스크립트의 `--camera 0`을 변경합니다.

Picamera2 포맷 이름은 메모리의 채널 순서와 직관적으로 반대로 보일 수 있습니다. OpenCV가 기대하는 BGR 배열을 받기 위해 `RGB888` 스트림을 사용하며, 모델 입력 직전에 BGR에서 RGB로 변환합니다.

## INT8 카메라 시각 시험

```bash
./02_run_int8_camera_test.sh
```

기본 설정:

| 항목 | 값 |
|---|---:|
| INT8 카메라 시험 | 600초 |
| ONNX Runtime CPU thread | 2 |
| 카메라 요청 크기 | 640×480 |
| 카메라 요청 FPS | 30 |
| 모델 입력 | 320×240 |
| 온도 자동 종료 | 75°C |

VNC 화면에 사람 박스, confidence, 추론시간, 전체 FPS와 CPU 온도가 표시됩니다. `Q`를 누르면 정상 종료되며 그 시점까지의 결과가 보존됩니다. `[`와 `]` 키로 confidence threshold를 0.05씩 조절할 수 있고 변경된 threshold도 프레임별 CSV에 기록됩니다.

영상 파일을 저장하지 않고도 장면별 결과를 구분할 수 있도록 OpenCV 카메라 창에 초점을 두고 다음 숫자 키를 누릅니다. 선택한 장면은 다음 프레임부터 CSV의 `scenario` 열에 기록됩니다.

| 키 | 장면 |
|---:|---|
| `0` | 사람이 없는 배경 |
| `1` | 가까운 사람 |
| `2` | 중간 거리 사람 |
| `3` | 먼 사람 |
| `4` | 움직이는 사람 |
| `5` | 여러 사람 |

영상과 카메라 이미지는 저장하지 않습니다. CSV와 JSON 숫자 결과만 저장합니다.

화면을 보면서 시험하려면 다음처럼 직접 실행할 수 있습니다.

```bash
.venv/bin/python pi_int8_camera_test.py --duration 600
```

화면 없이 순수 처리속도만 측정하려면 `--no-preview`를 추가합니다.

FP32와 INT8을 같은 조건으로 비교하는 추가 시험은 다음과 같습니다.

```bash
./03_run_fp32_int8_benchmark.sh
```

## 시험 장면

실제 RC카 카메라 높이와 각도에 가깝게 놓고 다음 장면을 골고루 포함합니다.

1. 사람이 없는 배경
2. 가까운 사람 1명
3. 중간 거리 사람 1명
4. 가능한 범위에서 먼 사람
5. 사람이 좌우로 이동하는 장면

두 모델의 공정한 비교를 위해 시험 중 카메라 위치와 조명을 가능한 한 유지합니다.

## 수집 결과

실행 후 다음 폴더가 생성됩니다.

```text
pi_int8_camera_results/YYYYMMDD_HHMMSS/
├── summary.json
├── frame_metrics.csv
└── detections.csv
```

### `summary.json`

- Raspberry Pi OS, CPU, RAM, Python, OpenCV, ONNX Runtime 버전
- 실제 카메라 해상도와 카메라가 보고한 FPS
- 평균/중앙값/P95 추론 지연시간과 카메라 포함 전체 FPS
- CPU 온도·주파수·사용률과 프로세스 CPU·RAM
- 카메라 timestamp부터 검출 결과가 만들어질 때까지 걸린 시간
- 노출시간, gain, Lux, 실제 프레임 주기
- 시험 시작/종료 throttling 상태
- confidence 및 사람 박스 크기 분포

### `frame_metrics.csv`

각 프레임의 다음 시간을 별도로 기록합니다.

- 카메라 캡처
- 전처리
- ONNX 추론
- 후처리
- 전체 반복
- CPU 사용률, RAM, 온도, 검출 개수

### `detections.csv`

검출된 사람마다 다음 값을 기록합니다.

- confidence와 프레임 번호
- 모델 입력 및 카메라 해상도 기준 bounding box
- 박스 중심점과 사람 발 위치에 가까운 bottom-center
- 정규화된 중심 및 bottom-center 좌표
- 사람 박스 너비·높이와 크기 구간

## 결과 해석 시 주의사항

- `model FPS`는 모델 추론만 기준으로 계산한 수치입니다.
- `end-to-end FPS`는 카메라 캡처, 전처리, 추론, 후처리를 포함합니다.
- P95 지연시간은 느린 쪽 5% 구간의 경계를 보여주므로 평균과 함께 봐야 합니다.
- `detection count agreement`는 두 모델의 상대적인 일치도이지 정답 기반 정확도가 아닙니다.
- 실제 mAP, Precision, Recall, F1은 정답 라벨이 있는 valid/test 데이터로 별도 평가해야 합니다.
- `vcgencmd get_throttled`가 `0x0`이 아니면 저전압 또는 온도 제한 기록을 해석해야 합니다.
- 현재 임시 모델 결과만으로 최종 FP32/INT8 배포 방식을 결정하지 않습니다.

## 추가 환경 정보

시험 결과와 함께 다음 출력도 기록합니다.

```bash
uname -a
cat /etc/os-release
vcgencmd get_throttled
```

최종 모델에서는 동일 시험을 반복한 뒤 정확도 손실, Raspberry Pi 전체 FPS, 발열 안정성을 함께 고려해 FP32 또는 INT8 배포 방식을 선택합니다.

## 2026-08-12 Raspberry Pi 4B 실측 결과

### 시험 환경

| 항목 | 확인값 |
|---|---|
| 보드 | Raspberry Pi 4 Model B Rev 1.5 |
| RAM | 4GB급, OS 확인값 3.7GiB |
| 운영체제 | Debian GNU/Linux 13 (trixie), 64-bit ARM |
| Kernel | Linux 6.18.34+rpt-rpi-v8 |
| CPU 논리 코어 | 4 |
| 카메라 | OV5647 CSI |
| 카메라 스트림 | 640×480, 약 30 FPS |
| 모델 입력 | 320×240 RGB, NCHW, float32 |
| OpenCV | 4.10.0 |
| ONNX Runtime | 1.27.0, CPUExecutionProvider |
| Python | 3.13.5 |
| 냉각 | 방열판 및 냉각팬 없음 |
| 시험 전 전원 상태 | `throttled=0x0` |

카메라는 Picamera2/libcamera를 통해 정상적으로 열렸으며 실제 프레임은 `(480, 640, 3)`, `uint8`로 확인했습니다. FP32와 INT8 ONNX 모델 모두 입력 `[1,3,240,320]`, 출력 `[1,5,15,20]`, 유한한 float32 출력을 생성했습니다.

### 색상 채널 오류와 수정

초기 시험에서는 Picamera2 스트림을 `BGR888`로 요청했고 얼굴과 피부가 파란색으로 표시됐습니다. Picamera2/libcamera의 포맷 이름과 Python 배열의 채널 순서는 직관과 다릅니다.

```text
잘못된 초기 경로:
Picamera2 BGR888 → Python RGB 배열 → OpenCV에 그대로 표시
                                  → 전처리에서 BGR→RGB 추가 변환

수정된 경로:
Picamera2 RGB888 → Python BGR 배열 → OpenCV 정상 표시
                                  → 모델 입력에서 BGR→RGB 변환
```

수정 후 피부색과 화면 색상이 정상으로 돌아왔고 사람 bounding box도 정상적으로 표시됐습니다. 따라서 색상 수정 전 시험은 속도와 발열 참고에는 사용할 수 있지만 검출 품질 판단에서는 제외합니다.

### CPU 스레드별 성능과 발열

같은 INT8 임시 모델을 이용해 ONNX Runtime CPU thread 수를 비교했습니다.

| 설정 | 평균 추론 | P95 추론 | 모델 FPS 환산 | 평균 camera-to-result | 최고 온도 | 종료 상태 |
|---|---:|---:|---:|---:|---:|---|
| 2 threads, 시험 1 | 90.22ms | 94.39ms | 11.1 FPS | 136.20ms | 71.58°C | 사용자 정상 종료 |
| 2 threads, 시험 2 | 89.95ms | 93.99ms | 11.1 FPS | 137.38ms | 68.65°C | 사용자 정상 종료 |
| 3 threads | 66.96ms | 75.17ms | 14.9 FPS | 115.13ms | 75.47°C | 75°C 안전 종료 |
| 4 threads | 62.06ms | 83.18ms | 16.1 FPS | 111.92ms | 78.39°C | 78°C 안전 종료 |

3 threads는 2 threads보다 빠르지만 냉각장치가 없는 상태에서 안전 온도에 도달했습니다. 4 threads는 3 threads보다 평균 추론시간이 약 4.9ms만 짧으면서 온도는 더 높아졌습니다.

따라서 현재 하드웨어의 지속 운용 기본값은 다음과 같습니다.

```text
ONNX Runtime provider : CPUExecutionProvider
intra-op threads      : 2
inter-op threads      : 1
confidence threshold  : 0.20
NMS IoU threshold     : 0.20
thermal stop          : 75°C
```

방열판이나 팬을 추가하면 3 threads를 다시 시험할 수 있지만, 냉각장치가 없는 현재 상태에서는 2 threads를 기본값으로 사용합니다. 모든 시험에서 `throttled=0x0`을 유지했습니다.

### 색상 수정 후 대표 유효 시험

대표 결과 폴더:

```text
pi_int8_camera_results/20260812_153248/
```

| 항목 | 결과 |
|---|---:|
| 실행시간 | 120초 |
| 처리 프레임 | 1,098 |
| 검출 발생 프레임 | 1,013 |
| 총 검출 박스 | 1,402 |
| 프레임당 검출 박스 | 1.28 |
| 평균 추론시간 | 90.63ms |
| P95 추론시간 | 95.65ms |
| 평균 전체 loop 시간 | 97.36ms |
| 평균 camera-to-result | 137.75ms |
| P95 camera-to-result | 155.92ms |
| 평균 온도 | 66.37°C |
| 최고 온도 | 72.55°C |
| 시작/종료 throttling | `0x0` / `0x0` |

Confidence 분포:

| 통계 | 값 |
|---|---:|
| 평균 | 0.3803 |
| 중앙값 | 0.3928 |
| P95 | 0.5748 |
| 최소 | 0.2020 |
| 최대 | 0.6309 |

이 실행에서는 장면 키를 누르지 않아 `scenario=unlabeled`로 저장됐습니다. 따라서 CSV만으로 거리별 검출률이나 빈 배경 오검출률을 계산할 수는 없습니다. 다만 VNC에서 사용자가 다음 사항을 직접 확인했습니다.

- 피부색과 카메라 색상 정상
- 한 명의 사람 bounding box 위치 정상
- 사람이 없는 배경에서 눈에 띄는 오검출 없음
- 화면 표시와 실행 종료 정상

이는 배포 경로와 시각적 동작에 대한 1차 통과 결과이지 정답 라벨 기반 정확도 검증은 아닙니다.

### 현재 판단

1. Raspberry Pi 4B에서 OV5647과 INT8 ONNX 모델을 함께 실행할 수 있습니다.
2. 냉각장치가 없을 때 2 threads가 속도와 온도의 안정적인 절충점입니다.
3. 약 11 FPS의 모델 처리와 약 138ms의 camera-to-result 지연을 확인했습니다.
4. Confidence `0.20`에서 가까운 사람과 화면 표시가 정상 작동했습니다.
5. 색상 순서가 검출 결과에 영향을 줄 수 있으므로 `RGB888` Picamera2 설정을 유지해야 합니다.
6. 현재 모델은 최종 DSConv+Residual+FPN+Anchor-free 모델이 아닌 임시 INT8 모델입니다.
7. 최종 모델 선정 후 ONNX FP32/INT8 변환과 동일한 Raspberry Pi 시험을 다시 수행해야 합니다.

### 아직 이 시험으로 확정할 수 없는 항목

- Precision, Recall, F1, mAP
- 작은 사람의 정답 기반 Recall
- 실제 거리별 검출 한계
- FP32 대비 INT8의 정답 기반 정확도 손실
- 장시간 실제 RC카 통합 운용의 온도
- C++ 카메라·지도 표시 코드와 통합했을 때의 전체 지연
- 사람과 로봇 사이의 실제 미터 단위 거리

위 항목은 최종 모델, 정답 라벨 데이터, 실제 거리 표식 및 C++ 통합 환경을 확보한 다음 별도로 검증합니다.

### 다음 단계

1. 학교 노트북 학습 결과 회수 및 실험별 `best.pt` 비교
2. 최종 구조와 checkpoint 선정
3. PyTorch → ONNX FP32 변환 및 수치 일치 검사
4. Static INT8 PTQ와 정답 기반 정확도 비교
5. 최종 ONNX INT8을 Raspberry Pi에서 2 threads로 재시험
6. 필요 시 냉각장치 추가 후 3 threads 재검토
7. 최종 C++ 카메라 파이프라인과 timestamp 계약으로 통합
