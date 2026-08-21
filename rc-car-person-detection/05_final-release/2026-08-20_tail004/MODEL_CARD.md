# Model Card — person_detector_tail004_int8.onnx

## Intended use

낮 시간 운동장에서 Raspberry Pi 기반 RC-Car 전방 카메라 영상의 사람을 검출하고, 각 사람의 bbox와 confidence를 상위 시스템에 전달한다. AI 결과는 지도 표시에 사용하며 모터를 직접 제어하지 않는다.

## Model lineage

```text
group-aware dataset v1
  → custom DSConv+Residual/FPN48 anchor-free detector
  → multi-resolution and seed search
  → R46 448×336 seed15 epoch25
  → selective QDQ PTQ
  → TAIL004
```

## Source validation metrics

| Metric | Value |
|---|---:|
| mAP50:95 | 0.261832 |
| AP50 | 0.561210 |
| AP75 | 0.210810 |
| Precision | 0.817398 |
| Recall | 0.412851 |
| F1 | 0.548610 |
| Tiny `<16 px` recall | 0.002104 |
| Small `16~32 px` recall | 0.155005 |

## Quantization preservation

120장 FP32 비교, confidence 0.25, NMS IoU 0.5 기준:

- FP32 detections: 523
- TAIL004 detections: 529
- matched: 512
- missed: 11
- extra: 17
- retain: 0.978967
- mean matched IoU: 0.974578

이 값은 TAIL004와 FP32의 출력 일치도이며 ground-truth 정확도가 아니다. 정확도 상한은 source R46에 의해 결정된다.

## Known limitations

- tiny person recall이 매우 낮아 먼 사람을 놓칠 수 있다.
- 사람이 부분적으로만 보이거나 심한 역광·가림이 있는 경우 별도 정량 검증이 부족하다.
- hard-negative 실험으로 특정 운동장 FP를 줄일 수 있음을 확인했지만 최종 TAIL004는 original R46 source를 사용한다.
- confidence 0.25는 시작값이며 실제 운동장 FP/FN 측정으로 조정할 수 있다.
- 모델 거리는 직접 출력하지 않는다. 거리·지도 좌표 변환은 calibration 파트의 역할이다.

## Provenance and integrity

- source checkpoint SHA-256: `9a9e761f282f35a91a64e491b71ff142085e95e0021e3e5114280cffbbcbf29c`
- final ONNX SHA-256: `230755c15376065bdfbcea44cbc9259d9691ec48319bc96bdbfb15c38b3e01be`
- raw datasets and calibration images are intentionally not redistributed in this repository.
