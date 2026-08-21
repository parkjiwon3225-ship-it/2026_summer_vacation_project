# 2026-08-12 캘리브레이션 팀 연동 자료

사람 탐지 결과를 실제 위치 계산으로 연결할 때 AI 팀과 캘리브레이션 팀이 공유해야 할 좌표·시간·유효성 규약입니다.

## 문서

- [`CALIBRATION_TEAM_INTEGRATION_GUIDE.md`](CALIBRATION_TEAM_INTEGRATION_GUIDE.md): 라이브 영상 연동, 프레임별 출력 규약, 팀별 역할
- [`PERSON_DISTANCE_ESTIMATION.md`](PERSON_DISTANCE_ESTIMATION.md): bbox와 카메라 파라미터를 이용한 사람 거리·지면 위치 추정 설계

## 핵심 구분

- AI 팀: 원본 프레임 기준 bbox, confidence, bottom-center, capture timestamp, 모델/threshold 정보 제공
- 캘리브레이션 팀: 카메라 내부·외부 파라미터와 자세 정보를 이용해 광선·지면 교차 및 로봇/지도 좌표 계산
- 부분 인체나 화면 경계에 닿는 bbox는 사람 존재 판단에는 쓸 수 있어도 지면 위치 계산에는 유효하지 않을 수 있음
