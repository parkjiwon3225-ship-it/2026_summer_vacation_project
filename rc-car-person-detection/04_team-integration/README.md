# 04. 팀 연동 및 전달

AI 개발·경량화 결과를 캘리브레이션 팀과 로봇 제어 개발자에게 전달하기 위한 과거 인계 자료와 최종 인터페이스 문서를 모았습니다.

| 날짜 | 폴더·문서 | 대상 | 설명 |
|---|---|---|---|
| 2026-08-12 | [`calibration-interface-guide`](2026-08-12_calibration-interface-guide/) | 캘리브레이션 팀 | live bbox, timestamp, bottom-center, 거리 추정의 역할 분담 |
| 2026-08-13 | [`results4-training-handoff`](2026-08-13_results4-training-handoff/) | 캘리브레이션·AI 팀 | 당시 최고 results.4 checkpoint와 이어학습 절차를 보존한 과거 인계본 |
| 2026-08-20 | [`최종 TAIL004 통합 계약`](../05_final-release/2026-08-20_tail004/INTEGRATION_CONTRACT.md) | 로봇 제어·통합 개발 | OpenCV C++ 입력, ONNX Runtime 전처리·후처리, 좌표·timestamp 출력 규격 |

현재 통합에는 과거 results.4가 아니라 [`05_final-release/2026-08-20_tail004`](../05_final-release/2026-08-20_tail004/)의 최종 모델과 문서를 사용합니다. 과거 인계본은 개발 과정과 이어학습 재현용으로만 남깁니다.
