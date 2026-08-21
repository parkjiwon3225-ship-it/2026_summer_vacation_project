# C++ OpenCV + ONNX Runtime 통합 계약

## 역할 분리

- OpenCV C++: 카메라 열기, 320×240 BGR 프레임 획득
- AI 전처리: BGR→RGB, letterbox, float32 NCHW
- ONNX Runtime C++: TAIL004 추론
- AI 후처리: threshold, NMS, 원본 좌표 복원

OpenCV DNN으로 모델을 실행하지 않는다. 최종 runtime은 ONNX Runtime C++ CPU provider다.

## 입력

```text
name  : images
type  : float32
shape : [1, 3, 336, 448]
order : RGB / NCHW
range : 0.0~1.0
pad   : RGB(114,114,114)
```

OpenCV 입력 프레임은 `320×240`, `CV_8UC3`, BGR인지 매 프레임 경로 초기화 시 확인한다. 320×240과 448×336은 모두 4:3이므로 정상 입력에서는 padding이 없고 scale은 1.4지만, 구현은 일반 letterbox metadata(`scale`, `pad_x`, `pad_y`)를 보존해야 한다.

## 출력

```text
boxes  : float32 [1, 12502, 4], model-input 좌표의 x1,y1,x2,y2
scores : float32 [1, 12502], class sigmoid × quality sigmoid
```

박스 decode는 ONNX 그래프 안에 포함되어 있다. 출력에 anchor decode를 다시 적용하지 않는다.

후처리 시작값:

```text
confidence threshold : 0.25
NMS IoU threshold    : 0.5
max detections       : 100
```

NMS 후 좌표를 letterbox 역변환하고 `0≤x≤320`, `0≤y≤240`으로 clamp한다.

## 스레드와 최신 프레임 정책

- camera thread는 `camera.read(frame)` 성공 직후 `steady_clock` timestamp와 `frame_id`를 기록한다.
- AI queue를 누적하지 않고 최신 프레임 한 장만 유지한다.
- inference thread는 처리 완료 후 가장 최신 프레임을 가져온다.
- ONNX Runtime 시작 설정은 intra-op 2 threads, inter-op 1 thread를 권장한다.
- 발열·전체 시스템 FPS를 측정한 뒤 intra-op thread 수만 조정한다.

## 결과 구조

```cpp
struct PersonDetection {
    int x1;
    int y1;
    int x2;
    int y2;
    float confidence;
};

struct DetectionResult {
    uint64_t frame_id;
    int64_t capture_timestamp_us;
    std::vector<PersonDetection> persons;
};
```

`capture_timestamp_us`는 카메라/GPS/IMU가 공유하는 `std::chrono::steady_clock` 기준 microseconds를 사용한다.

## 시작 시 필수 검증

1. 모델 SHA-256 확인
2. ONNX 입력 이름·shape·type 확인
3. 카메라 `cols=320`, `rows=240`, `type=CV_8UC3` 확인
4. 빈 장면과 한 사람 장면으로 BGR/RGB 확인
5. model 좌표를 원본 320×240으로 복원했는지 확인
6. 최신 프레임 정책에서 queue 지연이 누적되지 않는지 확인
