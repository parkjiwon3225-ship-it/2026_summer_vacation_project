# RC-Car Person Detector — Model Design V1

## 1. 목표

- RC카 전방 카메라에서 사람 한 클래스만 탐지한다.
- 기본 입력은 RGB `320×240` letterbox 이미지다.
- 완성된 YOLO, SSD, Faster R-CNN 모델을 가져오지 않는다.
- PyTorch의 기본 연산과 자동미분은 사용하되 backbone, FPN, detection head,
  target assignment, loss, decode, NMS, 학습 루프를 직접 구현한다.
- 정확도뿐 아니라 추론 지연시간, 파라미터 수, 연산량과 배포 가능성을 함께 평가한다.

## 2. 데이터 기준

- 데이터셋: `data/processed/v1_grouped`
- split: group-aware Train/Valid/Test = 약 80/10/10
- 이미지: 15,384장
- person bbox: 113,460개
- source-group leakage: 0
- 자동 무결성 검사: PASS, errors 0
- 저장 라벨은 `class cx cy width height` 정규화 형식이다.
- 이 라벨 형식은 모델 구조를 YOLO로 제한하지 않는다. DataLoader에서 픽셀 단위
  `xyxy`로 변환하고, 학습 시 anchor-free target으로 다시 할당한다.

## 3. 전체 구조

```text
320×240 RGB
    ↓
DSConv + Residual Backbone
    ↓
C2: 80×60, C3: 40×30, C4: 20×15, C5: 10×8
    ↓
Lightweight top-down FPN (64 channels)
    ↓
P2: 80×60, P3: 40×30, P4: 20×15, P5: 10×8
    ↓
Anchor-free decoupled detection head
    ↓
person score + quality score + l/t/r/b distances
    ↓
decode → score filtering → NMS → person boxes
```

초기 P3/P4/P5 target assignment 검사에서 표본 GT 741개 중 작은 사람 22개가 positive
위치를 받지 못했다. P2 추가 후 미할당이 8개(1.08%)로 감소했으므로 `P2: 80×60`을
V1 기본 모델에 포함한다. 남은 객체 대부분은 한 변이 약 2.5–4px인 sub-cell 객체다.

## 4. Backbone V1

### 4.1 기본 블록

`DSResidualBlock`:

1. `1×1 pointwise convolution`
2. `3×3 depthwise convolution`
3. `1×1 pointwise projection`
4. BatchNorm과 SiLU
5. stride가 1이고 입출력 채널이 같으면 residual addition

Downsampling 블록은 depthwise convolution의 stride를 2로 설정한다. 입출력 크기가
달라지는 블록에는 identity residual을 적용하지 않는다.

### 4.2 단계별 출력

| 단계 | 출력 크기 | 채널 | 반복 | 용도 |
|---|---:|---:|---:|---|
| Stem | 160×120 | 16 | 1 | 초기 특징 |
| Stage 1 | 80×60 | 24 | 2 | P2 후보 |
| Stage 2 | 40×30 | 48 | 3 | C3 |
| Stage 3 | 20×15 | 96 | 3 | C4 |
| Stage 4 | 10×8 | 160 | 2 | C5 |

초기 채널 수는 V1 기준값이다. 첫 overfit test 전에 파라미터 수와 feature-map 메모리를
계산하며, 장치 제약에 따라 width multiplier를 추가할 수 있다.

## 5. Lightweight FPN V1

- C2, C3, C4, C5를 각각 `1×1 convolution`으로 64채널에 맞춘다.
- C5를 C4의 정확한 spatial size로 nearest upsample한 뒤 더한다.
- 합쳐진 C4를 C3의 정확한 spatial size로 upsample한 뒤 더한다.
- 각 합성 출력에 `3×3 depthwise + 1×1 pointwise` refinement를 적용한다.
- 크기가 홀수인 feature map이 있으므로 `scale_factor=2` 대신 목표 tensor의
  `size`를 명시해 정렬 오류를 방지한다.

## 6. Anchor-free Detection Head V1

각 P3/P4/P5 위치에서 anchor 없이 다음 값을 예측한다.

- person classification logit: 1채널
- quality/centerness logit: 1채널
- bbox distances `(left, top, right, bottom)`: 4채널

Head는 classification/quality 분기와 regression 분기를 분리한다. 각 분기는 무거운
일반 convolution 대신 DSConv 블록 2개를 기본값으로 사용한다. 세 FPN 레벨 간 head
가중치 공유 여부는 구현 시 파라미터 수를 비교한 후 결정한다. V1 기본안은 공유다.

Regression 출력은 양수가 되어야 하므로 `exp`보다 수치적으로 안정적인 `softplus`를
초기 기본값으로 사용한다. 거리는 해당 feature level의 stride 단위로 해석한다.

## 7. Target Assignment V1

기본 방식은 FCOS 계열의 location-based assignment다.

1. 각 feature 위치의 이미지 좌표를 계산한다.
2. 위치가 GT bbox 내부에 있는지 검사한다.
3. GT 중심 주변의 center-sampling 영역만 positive 후보로 제한한다.
4. level별 regression range로 객체 크기를 P3/P4/P5에 분배한다.
5. 여러 GT가 겹치면 면적이 작은 GT를 우선한다.
6. positive 위치의 target을 `l/t/r/b`, class 1, centerness로 생성한다.

초기 regression range는 고정하지 않고 학습 전 bbox 통계로 산출한다. 작은 사람 비율이
높기 때문에 범용 FCOS의 기본 range를 그대로 복사하지 않는다.

## 8. Loss V1

- classification: sigmoid focal loss
- bbox regression: GIoU 또는 CIoU loss
- quality: binary cross entropy
- total: `L_cls + λ_box L_box + λ_quality L_quality`

가중치는 첫 overfit test에서 각 loss의 크기와 gradient를 확인한 후 고정한다. 임의의
최종값을 설계 문서에서 미리 확정하지 않는다.

## 9. Inference V1

1. 각 level의 bbox 거리 예측을 이미지 좌표 `xyxy`로 decode
2. `sigmoid(class) × sigmoid(quality)`로 최종 score 계산
3. 낮은 score 제거
4. 이미지 경계로 clip
5. 단일 클래스 NMS
6. letterbox padding과 scale을 역변환해 원본 좌표 복원

NMS는 우선 순수 PyTorch로 구현하고 정확성을 기준 구현과 비교한다. 배포 단계에서만
장치가 제공하는 최적화 연산으로 교체할 수 있다.

## 10. 학습 및 검증 순서

1. Dataset/DataLoader와 좌표 변환 단위 테스트
2. 모델 forward shape 테스트
3. target assignment 시각화
4. loss가 NaN 없이 계산되는지 검사
5. 8–32장 작은 표본에 overfit하여 파이프라인 검증
6. 전체 Train 학습, Valid로 선택
7. 구조와 임계값을 확정한 뒤 Test는 최종 1회 평가
8. small/medium/large person 성능을 별도로 보고
9. 실제 RC카 영상에서 지연시간과 실패 사례 측정

## 11. 평가 지표

- mAP50
- mAP50:95
- precision, recall
- `<16px`, `16–32px`, `32–96px`, `≥96px` 높이 구간별 recall
- 파라미터 수
- MACs/FLOPs
- 목표 장치에서 전처리 포함 end-to-end latency와 FPS

## 12. 고정 결정과 실험 항목

### 고정

- 완성된 detector를 사용하지 않는 custom implementation
- DSConv + Residual backbone
- lightweight FPN
- anchor-free head
- 입력 기본값 `320×240`
- group-aware V1 dataset
- 작은 사람 target 보존을 위한 P2 검출 레벨
- Train/Valid로 개발하고 Test는 최종 평가에만 사용

### 실험으로 남김

- P2 채널 축소를 통한 연산량 절감 여부
- width multiplier
- FPN 채널 48/64/80
- head weight sharing
- GIoU 대 CIoU
- center-sampling radius와 regression ranges
- augmentation 강도
- score/NMS threshold
