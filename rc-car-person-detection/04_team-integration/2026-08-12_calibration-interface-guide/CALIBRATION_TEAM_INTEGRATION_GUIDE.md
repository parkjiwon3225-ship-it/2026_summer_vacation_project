# 딥러닝–캘리브레이션 팀 연동 가이드

이 문서는 RC카 사람 탐지 딥러닝 파트의 현재 방향과, 최종 모델을 캘리브레이션 시스템에 연결할 때 필요한 입력·출력 규약을 공유하기 위한 문서입니다.

현재 캘리브레이션 교육용 프로그램은 녹화 영상과 기존 TorchScript 모델을 기준으로 작성되어 있습니다. 교육용 코드를 지금 즉시 변경하라는 의미가 아니라, 향후 라이브 카메라와 최종 경량 모델을 합칠 때 양쪽 팀이 같은 기준을 사용할 수 있도록 사전에 인터페이스를 정리한 것입니다.

## 1. 딥러닝 파트 진행 상황

YOLO 완제품을 그대로 사용하지 않고 RC카의 제한된 연산 성능을 고려한 사람 전용 경량 모델을 직접 구현하고 있습니다.

- 입력: RGB `320×240`
- 전처리: 종횡비를 유지하는 letterbox, 여백값 114, 픽셀값 0~1
- Backbone: DSConv + Residual
- Neck: Lightweight FPN
- Detection levels: P2, P3, P4, P5
- Head: Anchor-free
- 탐지 클래스: person 단일 클래스
- 기준 모델: 약 34.5만 parameters
- Loss: Focal classification + Quality BCE + GIoU box loss
- 작은 사람 탐지를 위해 P2 feature 사용

데이터는 같은 원본 이미지 변형이나 연속 영상 프레임이 서로 다른 split에 들어가지 않도록 source group 단위로 분리했습니다.

| Split | Images | Boxes |
|---|---:|---:|
| Train | 12,322 | 88,750 |
| Valid | 1,531 | 12,404 |
| Test | 1,531 | 12,306 |

- Group leakage: `0`
- 데이터 무결성 검사: `PASS`
- Test split은 최종 모델 선택 전까지 사용하지 않음

현재 기준 모델과 5개의 단일 요소 변형을 각각 100 epoch로 비교하는 단계입니다.

1. 낮은 learning rate
2. 경량 FPN
3. 모델 용량 증가
4. Box loss 강화
5. 작은 사람용 center sampling 확대

최종 모델은 mAP50:95뿐 아니라 Precision, Recall, F1, 작은 사람 Recall, 추론 속도, 메모리 사용량과 모델 크기를 함께 비교해 선택합니다.

## 2. 기존 교육 모델과 최종 모델의 차이

현재 캘리브레이션 교육 코드는 모델이 단일 grid tensor를 반환한다고 가정하고 다음 값을 직접 해석합니다.

```text
objectness, tx, ty, tw, th
```

새 경량 모델의 내부 출력은 다음과 같습니다.

```text
P2/P3/P4/P5 각각:
- class_logits
- quality_logits
- LTRB distances
```

따라서 학습 결과의 `best.pt`를 기존 `person_detector_cpu.pt` 대신 복사하는 것만으로는 작동하지 않습니다.

또한 학습 중 생성되는 `best.pt`는 optimizer와 epoch 등을 포함한 학습 체크포인트이고, `torch::jit::load()`로 직접 읽는 TorchScript 파일이 아닙니다.

딥러닝 파트에서 최종 모델 선정 후 다음 산출물을 별도로 제공할 예정입니다.

- CPU용 TorchScript 또는 ONNX 모델
- 모델과 동일한 letterbox 전처리
- 모델 전용 decode 및 NMS
- C++ 또는 Python 연동 예제
- 입력·출력 및 threshold 설정 문서
- 샘플 영상과 예상 탐지 결과

캘리브레이션 팀이 P2~P5 내부 출력을 직접 해석하지 않도록, 딥러닝 파트에서 후처리까지 담당하고 최종 Bounding Box를 전달하는 방향을 권장합니다.

## 3. 제안하는 프레임별 출력 규약

```cpp
struct PersonDetection {
    int track_id;          // 초기 탐지 전용 버전은 -1
    float confidence;

    float x1;
    float y1;
    float x2;
    float y2;

    float foot_x;
    float foot_y;
};

struct FrameDetections {
    uint64_t frame_id;
    int64_t capture_timestamp_ms;
    int frame_width;
    int frame_height;

    std::vector<PersonDetection> persons;
};
```

Bounding Box 좌표 정의:

```text
(x1, y1): 왼쪽 위
(x2, y2): 오른쪽 아래
```

지면에 닿는 사람 위치는 Bounding Box 아래쪽 중앙으로 제공합니다.

```text
foot_x = (x1 + x2) / 2
foot_y = y2
```

모든 좌표는 `320×240` 모델 입력 좌표가 아니라 letterbox의 scale과 padding을 역변환한 실제 카메라 프레임 픽셀 좌표로 제공할 예정입니다.

## 4. 영상 좌표계 통일

Homography 대응점을 선택한 영상과 사람 탐지 좌표를 출력하는 영상은 해상도, crop, 회전과 렌즈 왜곡보정 여부가 모두 같아야 합니다.

예를 들어 아래 두 방식을 섞으면 좌표가 일치하지 않습니다.

```text
Homography: 렌즈 왜곡보정 후 영상
사람 탐지: 왜곡보정 전 원본 영상
```

양쪽 팀은 다음 중 하나를 공통 좌표계로 정해야 합니다.

- 카메라 원본 프레임
- 렌즈 왜곡보정 후 프레임

위성지도 투영 정확도가 필요하고 카메라가 광각이라면, 렌즈 내부 파라미터를 구한 뒤 왜곡보정된 프레임을 공통 기준으로 사용하는 방향을 권장합니다.

## 5. 녹화 영상과 라이브 영상의 차이

라이브 시스템에서는 모델 정확도 외에도 프레임 지연과 센서 동기화가 중요합니다.

### 최신 프레임 우선

추론 속도가 카메라 FPS보다 느릴 때 모든 프레임을 큐에 쌓으면 오래된 화면을 뒤늦게 처리하게 됩니다.

```text
카메라 수신
→ 가장 최신 프레임 1장 유지
→ 추론
→ 오래된 프레임 폐기
```

### 촬영 시각 제공

`capture_timestamp_ms`는 추론이 끝난 시각이 아니라 카메라에서 프레임을 받은 시각이어야 합니다. 캘리브레이션 파트는 이 시각과 가장 가까운 GPS·IMU 데이터를 사용해야 합니다.

딥러닝 파트는 다음 진단 정보도 제공할 예정입니다.

- frame ID
- capture timestamp
- inference latency
- 실제 FPS

### 사람 추적

초기 연동은 프레임별 detection만으로 진행할 수 있습니다. 최종 라이브 시연에서 같은 사람의 지도 위치가 흔들리거나 중복 표시되면 IoU 기반 추적 또는 경량 tracker를 추가해 `track_id`를 제공할 예정입니다.

## 6. 움직이는 RC카와 Homography

고정 Homography 하나를 모든 프레임에 적용할 수 있는 조건은 다음과 같습니다.

- 카메라 위치가 고정됨
- 카메라 방향과 기울기가 변하지 않음
- 대상 지면이 같은 평면임

RC카가 이동하면 카메라 위치, 방향과 기울기가 변하므로 녹화된 고정 시점 영상에서 만든 Homography 하나를 전체 주행에 그대로 적용하기 어렵습니다.

실제 주행에서는 다음 정보 중 일부를 이용해 현재 프레임의 카메라 자세를 반영해야 합니다.

- RC카 GPS
- RC카 진행 방향 또는 heading
- IMU yaw/pitch/roll
- 카메라 내부 파라미터
- 카메라 장착 높이와 각도

딥러닝 파트는 각 사람의 원본 프레임 기준 `foot_x`, `foot_y`와 촬영 시각을 제공합니다. 해당 픽셀을 지도 또는 GPS 좌표로 변환하는 방법은 캘리브레이션 파트에서 결정해야 합니다.

## 7. 실제 RC카 영상 검증

최종 모델 선택 후 실제 RC카 카메라 영상으로 별도 검증할 예정입니다.

확인할 환경:

- 낮은 카메라 높이
- 주행 중 흔들림과 motion blur
- 역광, 그림자와 야간 환경
- 사람의 다리 가림
- 화면 가장자리 렌즈 왜곡
- 급격한 방향 전환

최종 통합 평가에는 다음 항목을 추가합니다.

- 실제 RC 영상 Recall
- 분당 false detection 수
- 실시간 FPS
- 프레임 촬영부터 결과 출력까지의 지연시간
- 발 좌표의 프레임 간 흔들림
- 위성지도 투영 위치 오차

실제 RC카 영상에서 기존 데이터와 성능 차이가 크면, 해당 영상을 추가로 라벨링해 최종 모델을 fine-tuning할 수 있습니다.

## 8. 팀별 담당 범위 제안

| 딥러닝 파트 | 캘리브레이션 파트 |
|---|---|
| 사람 탐지 모델 학습·경량화 | 카메라 내부·외부 파라미터 계산 |
| 모델 전처리와 decode/NMS | 렌즈 왜곡보정 기준 결정 |
| Bounding Box와 confidence | foot point의 지도 좌표 변환 |
| 원본 프레임 foot point | GPS·IMU·카메라 시간 동기화 |
| capture timestamp와 latency | 이동 중 카메라 자세 반영 |
| 필요 시 track ID | 지도 표시와 좌표 안정화 |
| TorchScript/ONNX와 연동 예제 | 최종 RC카 프로그램 통합 |

## 9. 캘리브레이션 팀 확인 요청

아래 항목이 결정되면 딥러닝 배포 어댑터를 그 기준으로 맞출 수 있습니다.

1. 최종 실행 장비와 운영체제
2. 최종 모델 형식: TorchScript, ONNX, OpenCV DNN 등
3. C++ 또는 Python 중 최종 통합 언어
4. 카메라 입력 방식: 장치 번호, RTSP, GStreamer, Pi Camera 등
5. 실제 카메라 해상도와 FPS
6. 렌즈 왜곡보정을 적용할 위치
7. 필요한 탐지 결과 자료형
8. 한 프레임에서 허용할 최대 사람 수
9. 필요한 confidence threshold 또는 조정 방식
10. 사람별 track ID 필요 여부
11. 움직이는 RC카에서 Homography 또는 좌표변환을 갱신하는 방법
12. GPS·IMU·카메라 timestamp 동기화 방법

## 10. 최종 연동 목표

```text
라이브 카메라
→ 동일한 영상 전처리
→ 경량 사람 탐지
→ 원본 프레임 Bounding Box
→ 사람 발 좌표 + capture timestamp
→ GPS·IMU와 시간 동기화
→ 현재 카메라 자세 기반 좌표변환
→ 위성지도에 사람 위치 표시
```

현재 딥러닝 학습 방향은 유지합니다. 캘리브레이션 팀의 위 확인 사항에 맞춰 최종 모델을 선택한 뒤 배포 모델, 후처리 코드와 라이브 연동 인터페이스를 제공할 예정입니다.
