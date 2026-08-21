# 2026-08-14 results14 ONNX Export

`results.14` FPN48 모델의 ONNX/INT8 변환 결과를 보관하는 폴더입니다.

## 모델

```text
models/
├─ results14_fpn48_fp32.onnx
├─ results14_fpn48_int8_qdq_minmax.onnx
└─ results14_fpn48_int8_qdq_percentile.onnx
```

## 용도

- `FP32`: 정확도 비교 기준
- `INT8 QDQ MinMax`: MinMax calibration 기반 INT8 후보
- `INT8 QDQ Percentile`: Percentile calibration 기반 INT8 후보

`results.14`는 validation 기준 상위 checkpoint 계열이지만,
이 ONNX 변환 결과만으로 최종 Raspberry Pi 배포 모델을 확정하지 않습니다.

최종 선택 시 함께 비교할 항목:

- Precision / Recall / F1 / mAP
- tiny / small person recall
- False Positive / False Negative
- Raspberry Pi inference latency
- End-to-End FPS
- CPU / RAM
- 발열

다음 카메라 검증 작업:

[`../2026-08-16_results20-camera-validation`](../2026-08-16_results20-camera-validation/)
