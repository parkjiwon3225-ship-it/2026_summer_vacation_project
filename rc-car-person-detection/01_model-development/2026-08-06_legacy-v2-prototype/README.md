# 2026-08-06 초기 Person Detector v2 — Legacy

프로젝트 초기에 선생님이 제공한 결과 코드를 바탕으로 사용한 학습 노트북, PyTorch 체크포인트, TorchScript 변환 코드입니다. 이후 프로젝트 요구에 맞춰 DSConv+Residual, 경량 FPN, anchor-free head를 직접 구현했으므로 이 폴더는 개발 이력 보존용입니다.

## 포함 내용

- `person_detector_training_*.ipynb`: 초기 학습 노트북
- `person_detector_v2_best.pth`: 당시 최고 체크포인트
- `person_detector_v2_last.pth`: 당시 마지막 체크포인트
- `person_detector_v2_best_script.pt`: TorchScript 변환 결과
- `convert_v2_pth_to_pt.py`: 변환 스크립트

## 사용 시 주의

- 현재 추천 모델이 아닙니다.
- 최신 실험 결과와 직접 비교하지 않습니다.
- 현재 코드는 상위 폴더의 `2026-08-12_custom-anchor-free-detector`를 사용합니다.
