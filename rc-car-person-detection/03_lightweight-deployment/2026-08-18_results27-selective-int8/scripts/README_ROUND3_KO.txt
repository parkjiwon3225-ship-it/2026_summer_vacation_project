RESULTS.27 INT8 ROUND 3 — SELECTIVE / MIXED PRECISION

목표
- Round 1/2에서 모든 Conv를 INT8로 만들었을 때 검출 보존율이 크게 무너졌습니다.
- Round 3에서는 detector head를 FP32로 남기고 연산량이 큰 앞단만 INT8화합니다.

파일
06_inspect_conv_groups.py
06_quantize_int8_round3.py
07_validate_int8_round3.py

설치 위치
현재 results27_lightweight_pipeline\scripts\ 폴더에 세 파일을 복사합니다.

실행 순서
1) 노드 그룹 확인
python scripts\06_inspect_conv_groups.py

2) selective INT8 생성
python scripts\06_quantize_int8_round3.py --calibration-samples 160

3) 80장 보존성 검증
python scripts\07_validate_int8_round3.py --samples 80 --threads 2

후보
- backbone_only : backbone Conv만 INT8
- backbone_fpn  : backbone + FPN Conv INT8, head FP32
- no_head       : head를 제외한 모든 Conv INT8

양자화 기본값
- QDQ
- activation QInt8
- weight QInt8
- per-channel
- Percentile 99.99
- calibration 160 images

주의
- 개인 노트북 x86 latency는 Raspberry Pi ARM 성능의 대체값이 아닙니다.
- 최종 채택은 FP32 대비 보존성 + Pi latency + 실제 학교 운동장 FP/FN으로 결정합니다.
