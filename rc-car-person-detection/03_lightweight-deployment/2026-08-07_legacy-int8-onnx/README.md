# Person Detector v2 INT8 ONNX — 선생님 제공 코드 기반 Legacy

> 이 폴더는 선생님이 제공한 결과 코드와 초기 v2 모델을 바탕으로 2026-08-07에 수행한 임시 경량화 이력입니다. 우리 팀이 이후 직접 설계한 최종 후보가 아닙니다. 전체 INT8 발전 과정은 상위 폴더의 `2026-08-13_int8-model-evolution-and-finalists`, 현재 Raspberry Pi 시험은 `2026-08-13_results4-pi-model-variants`를 사용하세요.

## Description

Person detection model converted from PyTorch to ONNX
and quantized to INT8 format.

## Model Files

### FP32 Model

- person_detector_v2_best.onnx

Original ONNX model.

### INT8 Model

- person_detector_v2_best_int8.onnx

Quantized INT8 model for lightweight inference.

## Conversion Pipeline


## Test Result

ONNX Runtime inference test completed successfully.

Input:


[batch, 3, 240, 320]


Output:


(1, 5, 5, 30, 40)


## Target Device

- Raspberry Pi
- OpenCV camera detection system
