# Raspberry Pi 1차 FP32/INT8 테스트

이 폴더의 두 모델은 최종 학습 모델이 아니라 Raspberry Pi 배포 경로와 성능을 먼저 확인하기 위한 임시 모델입니다.

## 측정하는 항목

- FP32/INT8 모델 로딩 시간과 파일 크기
- 카메라 캡처, 전처리, 모델 추론, 후처리, 전체 반복 시간
- 평균/중앙값/P95 지연시간과 실제 전체 FPS
- CPU 사용률, 프로세스 RAM, CPU 온도, throttling 상태
- 동일 프레임에서 FP32/INT8 검출 개수 일치율
- 일치 박스의 IoU와 confidence 차이

영상과 카메라 이미지는 저장하지 않습니다. CSV와 JSON 숫자 결과만 저장합니다.

## 준비

Raspberry Pi에 이 폴더 전체를 복사합니다. 두 ONNX 파일도 이 문서와 같은 폴더에 있어야 합니다.

```text
person_detector_fp32.onnx
person_detector_int8.onnx
pi_first_benchmark.py
requirements-pi.txt
01_setup_pi.sh
02_run_pi_test.sh
```

터미널에서 폴더로 이동한 뒤 최초 한 번 실행합니다.

```bash
chmod +x 01_setup_pi.sh 02_run_pi_test.sh
./01_setup_pi.sh
```

마지막에 `SETUP PASS`가 나오는지 확인합니다.

## 카메라 확인

```bash
ls -l /dev/video*
```

OV5647 CSI 카메라는 `/dev/video0`을 OpenCV로 직접 여는 대신 Picamera2/libcamera를 사용합니다. `rpicam-hello --list-cameras`에서 카메라 번호가 `0`이면 기본 설정을 그대로 사용합니다.

## 시험 환경

카메라에 다음 장면이 시험 중 골고루 들어오게 합니다.

1. 가까운 사람 1명
2. 중간 거리 사람 1명
3. 가능한 범위에서 먼 사람
4. 사람이 좌우로 이동하는 장면
5. 사람이 없는 배경

조명과 카메라 각도는 실제 RC카 사용 조건에 가깝게 유지합니다. 테스트 중 다른 무거운 프로그램은 실행하지 않습니다.

## 실행

```bash
./02_run_pi_test.sh
```

기본 시험은 FP32 3분, INT8 3분, 동일 프레임 비교 300장으로 진행됩니다. 화면을 보면서 확인하려면 마지막 명령에 `--preview`를 추가할 수 있지만, 순수 성능 측정에는 preview를 사용하지 않는 편이 좋습니다.

## 보내줄 결과

완료 후 화면에 다음 형식의 경로가 표시됩니다.

```text
pi_test_results/날짜_시간/
```

그 폴더 전체를 가져오면 됩니다.

- `summary.json`: 환경과 핵심 요약
- `frame_metrics.csv`: 프레임별 속도·온도·메모리
- `comparison.csv`: FP32/INT8 검출 차이

추가로 터미널에서 아래 명령 결과도 사진으로 남깁니다.

```bash
uname -a
cat /etc/os-release
vcgencmd get_throttled
```

`get_throttled=0x0`이면 현재와 과거에 저전압 또는 thermal throttling이 기록되지 않았다는 의미입니다. 0이 아니면 해당 값을 그대로 전달합니다.

## 주의

- 이번 검출 일치율은 FP32와 INT8의 상대 비교이며 실제 정답 기반 mAP/Recall이 아닙니다.
- 최종 모델이 완성되면 동일한 시험을 다시 실행합니다.
- 최종 채택은 정확도 평가와 Raspberry Pi 실측을 함께 보고 결정합니다.
