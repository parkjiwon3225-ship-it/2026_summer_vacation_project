# RC-Car 사람 거리 및 지면 위치 추정 설계

## 1. 문서 목적

이 문서는 RC-Car Person Detection 모델이 검출한 사람의 bounding box를 이용해 다음 값을 계산하는 방법을 정의합니다.

- 로봇 기준 사람의 전방·좌우 위치
- 로봇과 사람 사이의 지면 거리
- 로봇 기준 사람의 방위각
- 이후 지도 표시에서 사용할 사람 위치 정보

OpenCV나 사람 검출 모델은 이미지 안의 bounding box를 제공할 뿐 실제 거리 단위를 직접 출력하지 않습니다. 실제 거리 계산에는 카메라 캘리브레이션, 카메라 설치 위치, pan/tilt, IMU 자세 및 timestamp 정렬이 필요합니다.

## 2. 권장 계산 방식

권장 방식은 사람 bounding box의 아래쪽 중앙을 사람의 발이 지면에 닿는 지점으로 보고, 해당 픽셀에서 생성한 카메라 광선과 지면 평면의 교차점을 계산하는 것입니다.

```text
카메라 프레임
→ Person Detector bounding box
→ bottom-center 픽셀 선택
→ 렌즈 왜곡 제거
→ 픽셀을 카메라 광선으로 변환
→ 카메라 설치각과 pan/tilt 반영
→ IMU roll/pitch 반영
→ 광선과 지면 평면의 교차점 계산
→ 로봇 기준 위치·거리·방위각 계산
→ 필요한 경우 지도 좌표로 변환
```

카메라가 pan/tilt로 움직이기 때문에 하나의 고정 pixel-to-ground homography를 모든 프레임에 적용해서는 안 됩니다. 매 프레임의 pan/tilt 각도를 이용해 카메라 외부 파라미터 또는 지면 homography를 갱신해야 합니다.

## 3. 좌표계 정의

통합 전에 세 팀이 동일한 좌표축과 각도 부호를 사용해야 합니다. 다음 정의를 권장합니다.

### 카메라 좌표계

```text
+Xc : 이미지 오른쪽
+Yc : 이미지 아래쪽
+Zc : 카메라가 바라보는 정면
```

### 로봇 좌표계

```text
+Xr : RC카 전방
+Yr : RC카 왼쪽
+Zr : 위쪽
원점: 로봇 기준점 또는 회전 중심
```

카메라와 로봇 좌표계 정의가 다르므로 고정 회전 행렬 `R_mount`와 카메라 원점의 이동 벡터 `t_camera_robot`이 필요합니다.

pan, tilt, IMU 각도는 다음 항목까지 명시해야 합니다.

- 단위: degree 또는 radian
- 양의 회전 방향
- 영점의 실제 방향
- 값의 범위
- timestamp 단위와 기준 clock

## 4. Person Detector에서 사용하는 픽셀

검출된 사람 박스가 다음과 같다고 가정합니다.

```text
xmin, ymin, xmax, ymax
```

사람의 지면 접점으로 사용할 픽셀은 다음과 같습니다.

```text
u = (xmin + xmax) / 2
v = ymax
```

이를 `bottom-center`라고 합니다.

박스 중심은 사람 몸통 부근의 공중 위치이므로 지면 교차에 적합하지 않습니다. 발이 프레임 밖으로 잘렸거나 다른 물체에 가려진 경우에는 bottom-center의 신뢰도가 낮으므로 거리 결과를 무효화하거나 낮은 위치 신뢰도를 함께 전달해야 합니다.

## 5. 카메라 내부 파라미터

캘리브레이션 팀에서 다음 값을 제공해야 합니다.

```text
fx, fy       : 픽셀 단위 초점거리
cx, cy       : 광학 중심
distCoeffs   : 렌즈 왜곡 계수
image_width  : 캘리브레이션 기준 이미지 너비
image_height : 캘리브레이션 기준 이미지 높이
```

카메라 행렬은 다음과 같습니다.

```text
K = [ fx   0  cx ]
    [  0  fy  cy ]
    [  0   0   1 ]
```

카메라 캘리브레이션 해상도와 실제 추론 입력 또는 카메라 캡처 해상도가 다르면 내부 파라미터를 동일한 해상도에 맞게 스케일링해야 합니다.

OpenCV C++에서는 `cv::undistortPoints()`를 이용해 bottom-center의 렌즈 왜곡을 제거할 수 있습니다.

왜곡이 제거된 픽셀로부터 카메라 좌표계 광선을 생성합니다.

```text
ray_camera = normalize([
    (u - cx) / fx,
    (v - cy) / fy,
    1
])
```

## 6. 카메라 외부 파라미터와 pan/tilt

필요한 고정 설치 측정값은 다음과 같습니다.

- 지면에서 카메라 광학 중심까지의 높이
- 로봇 기준점에서 카메라 광학 중심까지의 X/Y/Z 이동
- pan=0, tilt=0일 때 카메라 광학축 방향
- pan축과 tilt축의 실제 회전 방향
- 서보 명령각과 실제 카메라 각도의 오프셋 및 비선형 오차

프레임 시각의 카메라 광선을 로봇 좌표계로 변환하는 기본 구조는 다음과 같습니다.

```text
ray_robot =
    R_imu
    × R_mount
    × R_pan(pan)
    × R_tilt(tilt)
    × ray_camera
```

실제 행렬 곱 순서는 좌표계 정의와 pan/tilt 기구의 부모-자식 회전축 순서에 맞춰 검증해야 합니다. 위 식을 그대로 복사하기보다 실제 기구 구조에 맞춰 단위시험을 해야 합니다.

카메라의 로봇 좌표계 원점은 다음과 같습니다.

```text
origin_robot = [camera_x, camera_y, camera_z]
```

## 7. IMU 자세 보정

평평하지 않은 지면, 급가속 및 차체 흔들림은 카메라가 지면을 바라보는 각도를 바꿉니다.

- roll: 차체 좌우 기울기
- pitch: 차체 앞뒤 기울기
- yaw: 세계 좌표에 표시할 때 필요한 진행 방향

로봇과 사람 사이의 상대 위치만 계산할 때는 roll과 pitch가 가장 중요합니다. 사람 위치를 지도상의 절대 좌표로 변환할 때는 yaw 또는 별도의 heading 정보가 필요합니다.

IMU 원시값에는 진동과 순간 오차가 있으므로 시간 보간과 적절한 필터링이 필요하지만, 필터가 지나치게 느리면 이동 중 과거 자세를 적용하는 문제가 생길 수 있습니다.

## 8. 광선과 지면의 교차

로봇 좌표계에서 지면 평면을 다음과 같이 정의합니다.

```text
normalᵀ × point + d = 0
```

카메라 원점에서 사람의 발 방향으로 나가는 광선은 다음과 같습니다.

```text
point(lambda) = origin_robot + lambda × ray_robot
```

지면과 만나는 스케일은 다음과 같습니다.

```text
lambda = -(normalᵀ × origin_robot + d)
         / (normalᵀ × ray_robot)
```

사람의 로봇 기준 지면 위치는 다음과 같습니다.

```text
person_robot = origin_robot + lambda × ray_robot
```

다음 경우에는 유효한 교차로 처리하지 않습니다.

- 분모가 0에 가까워 광선이 지면과 거의 평행함
- `lambda <= 0`으로 교차점이 카메라 뒤쪽에 있음
- 계산된 거리가 설정한 유효 범위를 벗어남
- 사람의 발 부분이 프레임 밖으로 잘림
- 박스가 너무 작거나 confidence가 낮음
- pan/tilt 또는 IMU 데이터가 프레임 timestamp와 충분히 가깝지 않음

## 9. 로봇 기준 거리와 방향

사람 위치가 다음과 같다면:

```text
person_robot = [x_robot, y_robot, z_robot]
```

로봇과 사람 사이의 지면 거리는 다음과 같습니다.

```text
distance_m = sqrt(x_robot² + y_robot²)
```

로봇 정면 기준 방위각은 좌표축 정의에 맞춰 다음과 같이 계산할 수 있습니다.

```text
bearing_rad = atan2(y_robot, x_robot)
bearing_deg = bearing_rad × 180 / pi
```

카메라 기준 거리가 아니라 로봇 기준 거리로 만들려면 카메라와 로봇 원점 사이의 이동 벡터를 반드시 반영해야 합니다.

## 10. 단순 높이·각도 방식

평평한 지면에서 카메라 높이가 `h`, 사람의 발을 내려다보는 각도가 `alpha`라면 다음 근사식도 사용할 수 있습니다.

```text
distance ≈ h / tan(alpha)
```

이 식은 개념 확인에는 유용하지만 실제 구현에서는 렌즈 왜곡, 카메라 위치, pan/tilt, IMU와 3차원 좌표 변환을 반영한 광선-평면 교차 방식을 권장합니다.

사람의 실제 키를 가정하는 방식도 있습니다.

```text
distance ≈ fy × assumed_person_height / person_pixel_height
```

하지만 사람마다 키와 자세가 다르고 발이나 머리가 가려질 수 있으므로 주 거리 계산 방식으로 사용하지 않습니다. 지면 교차가 불가능할 때의 낮은 신뢰도 보조값으로만 고려합니다.

## 11. 낮은 카메라 높이에 따른 한계

현재 RC카 카메라는 지면에서 매우 낮은 위치에 장착됩니다. 카메라 높이가 낮으면 먼 거리의 사람에 대한 지면 광선 각도가 수평선에 매우 가까워져 작은 픽셀·각도 오차가 큰 거리 오차로 확대됩니다.

예를 들어 카메라 높이가 0.1m이고 사람이 5m 떨어져 있으면 지면 하향각은 약 1.15도입니다. 이 영역에서 tilt, IMU 또는 bottom-center가 1도 수준으로 어긋나면 계산 거리가 수 m 이상 달라질 수 있습니다.

따라서 다음 정책이 필요합니다.

- 유효 거리 상한 설정
- 지평선에 가까운 광선 무효 처리
- 작은 사람 박스에 낮은 위치 신뢰도 부여
- 여러 프레임의 위치를 이용한 outlier 제거와 완만한 필터링
- 알려진 실제 거리별 현장 보정식 적용
- 가능하다면 카메라 설치 높이 증가 검토

## 12. timestamp 정렬

카메라, pan/tilt, IMU와 로봇 위치 timestamp가 서로 다르므로 가장 최근 값을 단순 결합하면 안 됩니다.

Person Detection 결과에는 원본 카메라 프레임의 촬영 timestamp를 유지해야 합니다. 이 timestamp를 기준으로 다음 값을 보간합니다.

```text
pan(frame_time)
tilt(frame_time)
roll(frame_time)
pitch(frame_time)
heading(frame_time)
robot_position(frame_time)
```

카메라가 회전하거나 RC카가 움직이는 동안 현재 센서값을 과거 프레임에 적용하면 사람 방향과 지도 위치가 틀어집니다.

필요한 timestamp 계약:

- 모든 timestamp의 clock 기준
- 단위: 권장 nanosecond
- monotonic clock과 wall clock 사용 구분
- 센서별 측정 시각인지 수신 시각인지 구분
- 허용할 최대 시간 차이
- 보간 실패 시 결과 무효 처리 규칙

## 13. 지도 좌표 변환

로봇 기준 사람 위치를 지도 좌표로 표시하려면 같은 프레임 시각의 로봇 위치와 heading을 사용합니다.

```text
person_world =
    robot_world_position
    + R_world_robot(heading) × person_robot
```

GPS는 로봇과 사람 사이의 상대거리 계산에는 필요하지 않습니다. 로봇 기준으로 계산된 사람 위치를 세계 또는 지도 좌표로 옮길 때 사용합니다.

GPS 자체 오차가 사람 거리 추정 오차보다 클 수 있으므로 상대 위치 정확도와 지도 절대 위치 정확도를 분리해 평가해야 합니다.

## 14. C++ 전달 데이터 구조 제안

AI, 캘리브레이션 및 로봇 제어 모듈 사이에서는 최소한 다음 값을 전달하는 것을 권장합니다.

```cpp
struct PersonGroundPosition {
    uint64_t frame_timestamp_ns;

    float detection_confidence;

    float bbox_xmin;
    float bbox_ymin;
    float bbox_xmax;
    float bbox_ymax;

    float bottom_center_u;
    float bottom_center_v;

    float robot_x_m;
    float robot_y_m;
    float distance_m;
    float bearing_deg;

    float position_confidence;
    bool ground_intersection_valid;
};
```

`detection_confidence`와 `position_confidence`는 서로 다른 값입니다.

- `detection_confidence`: 이미지에 사람이 존재한다는 AI 신뢰도
- `position_confidence`: 해당 사람의 지면 위치와 거리가 믿을 만한 정도

사람을 잘 검출해도 발이 가려졌거나 지평선에 가까우면 위치 신뢰도는 낮을 수 있습니다.

## 15. 팀별 책임 범위

### AI 개발 및 경량화 팀

- bounding box와 detection confidence 제공
- 원본 카메라 frame timestamp 유지
- bottom-center 및 박스 크기 제공
- 발 잘림, 작은 박스 등 위치 계산 품질 플래그 제공
- Raspberry Pi 추론 지연시간 기록

### 캘리브레이션 팀

- 카메라 내부 파라미터와 왜곡 계수 제공
- 카메라-로봇 외부 파라미터 정의
- pan/tilt 각도 모델과 영점 보정
- IMU 자세를 반영한 광선-지면 교차 구현
- 위치 신뢰도 및 유효 거리 범위 정의
- 알려진 거리 기반 실측 오차 평가와 보정

### 로봇 제어 팀

- frame timestamp와 동일 기준의 pan/tilt 제공
- 동일 기준의 IMU roll/pitch 및 heading 제공
- 로봇 좌표계와 각도 부호 정의 공유
- 지도 표시가 필요할 때 로봇 위치와 heading 제공

AI 결과는 모터 제어에 사용하지 않고 사람 위치 지도 표시에 사용합니다.

## 16. 현장 검증 계획

알려진 실제 거리에 사람을 세우고 다음 조건을 반복 측정합니다.

```text
거리: 0.5m, 1m, 2m, 3m, 5m 또는 실제 사용 범위
방향: 정면, 좌측, 우측
pan/tilt: 여러 대표 각도
차체 자세: 평지, 작은 roll/pitch 변화
사람 상태: 정지, 이동, 일부 가림
```

각 조건에서 다음을 기록합니다.

- 실제 거리와 추정 거리
- 절대 오차와 상대 오차
- 실제 방위각과 추정 방위각
- bounding box confidence와 크기
- bottom-center 흔들림
- pan/tilt 및 IMU 값
- 프레임 timestamp와 센서 timestamp 차이

거리별 오차 평균뿐 아니라 P95 오차와 실패율을 확인해야 합니다. 최종적으로 허용 가능한 거리 범위와 `position_confidence` 규칙을 결정합니다.

## 17. 최종 설계 요약

우리 프로젝트에서 권장하는 거리 계산 방식은 다음과 같습니다.

```text
Person bounding-box bottom-center
→ 렌즈 왜곡 보정
→ 카메라 광선 생성
→ 카메라 설치각 + pan/tilt + IMU 자세 반영
→ 지면 평면과 교차
→ 로봇 기준 X/Y, 거리, 방위각 계산
→ 신뢰도 및 유효성 검사
→ 필요할 때 GPS/heading으로 지도 좌표 변환
```

낮은 카메라 높이와 움직이는 pan/tilt 때문에 먼 거리에서는 오차가 빠르게 커질 수 있습니다. 따라서 수학적 계산만으로 완료했다고 판단하지 않고 실제 거리별 현장 검증을 통해 유효 범위와 보정식을 확정해야 합니다.
