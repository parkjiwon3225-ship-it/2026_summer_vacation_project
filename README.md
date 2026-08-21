# 2026 Summer Vacation Projects

2026년 여름방학 동안 진행한 개발 프로젝트와 실험 기록을 정리하는 개인 저장소입니다.

## RC Car Person Detection

Raspberry Pi 4 + PiCar-X 기반 RC카에서 실시간으로 사람을 탐지하기 위한 경량 객체 탐지 모델 개발 프로젝트입니다.

- 프로젝트 폴더: [`rc-car-person-detection/`](./rc-car-person-detection/)
- 원본 협업 저장소: `GGulBe/rc-car-project`
- 원본 브랜치: `ai-model`
- 기준 원본 커밋: `76c814c5ee31d6903c48ba61b9ebbc59e7d55047`
- 최종 선택 모델: Tail004
- 개발 흐름: 데이터 분석 → 모델 설계 → 반복 학습/비교 → ONNX/INT8 경량화 → Raspberry Pi 실차 벤치마크 → 최종 모델 선정

세부 실험 과정, 비교 결과, 배포 자료 및 최종 릴리스는 프로젝트 폴더의 README와 각 단계별 디렉터리에 정리되어 있습니다.
