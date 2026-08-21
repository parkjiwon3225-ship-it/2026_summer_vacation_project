# Round 3 30-epoch 선별 결과와 연휴 장시간 학습 계획

기준일: 2026-08-14

데이터: `data/processed/v1_grouped`

평가 split: Valid

Test split: 사용하지 않음

## 목적

학교 노트북 6대에서 FPN48 중심 후보를 약 30 epoch까지 학습해 어떤 설정을 연휴 장시간 학습에 사용할지 결정했다. 이 실험은 최종 모델 완성이 아니라 expansion, box loss weight, center sampling radius와 seed 방향을 좁히기 위한 선별 실험이다.

## 공통 조건

```text
Input                   320x240
Planned epochs          100
Batch                   8
Learning rate           0.001
Weight decay            0.0001
FPN channels            48
Quality loss weight     1.0
AMP                     on
AMP initial scale       1024
AMP growth interval     20000
Best checkpoint metric  Valid mAP50:95
Operating threshold     0.25
NMS IoU                 0.50
```

## 결과

| 순위 | 자료 | 변경점 | 완료 epoch | best epoch | mAP50:95 | AP50 | AP75 | P@0.25 | R@0.25 | F1@0.25 | tiny R | small R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | results.14 | exp2.0 · box2.0 · radius1.5 · seed11 | 30 | 29 | **0.253238** | 0.550494 | 0.201683 | 0.673 | **0.510** | **0.581** | **0.074** | **0.453** |
| 2 | results.15 | 기준 · seed14 | 31 | 20 | 0.230929 | 0.498915 | 0.179969 | **0.854** | 0.243 | 0.378 | 0.002 | 0.185 |
| 3 | results.19 | radius 2.0 · seed11 | 28 | 28 | 0.228456 | 0.502602 | 0.177840 | 0.793 | 0.353 | 0.489 | 0.012 | 0.209 |
| 4 | results.16 | expansion 2.5 · seed11 | 30 | 16 | 0.220541 | 0.507530 | 0.158066 | 0.790 | 0.359 | 0.494 | 0.024 | 0.243 |
| 5 | results.17 | expansion 2.5 · seed14 | 31 | 27 | 0.220267 | 0.485520 | 0.170968 | 0.827 | 0.255 | 0.390 | 0.004 | 0.147 |
| 6 | results.18 | box weight 2.5 · seed11 | 32 | 30 | 0.209304 | 0.478252 | 0.154821 | 0.799 | 0.311 | 0.447 | 0.011 | 0.179 |

모든 실행이 도달한 epoch 28까지 잘라 비교해도 판단은 바뀌지 않았다.

- expansion 2.0 두 실행의 epoch≤28 최고 mAP 평균: `0.240682`
- expansion 2.5 두 실행의 epoch≤28 최고 mAP 평균: `0.220404`
- expansion 2.0 우위: `+0.020278`
- results.14의 epoch 20~28 평균 mAP: `0.2421`
- results.14의 이전 results.4 대비 최고 mAP 차이: `+0.000760`

`results.14`는 수치상 새 최고이지만 results.4의 `0.252478`과 차이가 작다. 큰 개선으로 선언하기보다 FPN48/expansion2.0 계열이 약 0.25를 다시 재현했다는 근거로 해석한다.

## 변수별 결정

| 변수 | 관찰 | 결정 |
|---|---|---|
| Expansion 2.5 | 두 seed 평균에서 expansion 2.0보다 약 0.020 mAP 낮고 checkpoint도 더 큼 | 장시간 주력에서 제외 |
| Box weight 2.5 | 동일 seed 기준 mAP 약 0.044 하락, valid loss 1.822 | 제외, 2.0 유지 |
| Radius 2.0 | 마지막까지 상승했으나 tiny/small Recall이 0.012/0.209로 낮음 | 제외, 1.5 유지 |
| Seed | 동일 구조에서도 결과 차이가 큼 | 320과 고해상도 경로에서 복수 seed 유지 |

최종 장시간 공통 구조는 다음으로 고정했다.

```text
FPN channels            48
Backbone expansion      2.0
Box loss weight         2.0
Quality loss weight     1.0
Center sampling radius  1.5
Final input             320x240
```

## 안정성과 중단 원인

- 6개 실행 모두 warning `0`
- NaN/Inf gradient epoch `0`
- traceback·CUDA OOM 없음
- runner log의 종료 원인은 `User interrupted training`
- 각 실험의 `best.pt`, `last.pt`, `history.csv` 정상 보존

따라서 오류로 멈춘 결과가 아니라, 약 30 epoch에서 비교하기 위해 사용자가 중단한 정상 선별 결과다.

## 연휴 장시간 6대 배치

| 노트북 | 단계 | 목적 |
|---|---|---|
| 학교 1 | 320x240 · seed11 · 200 epoch | 현재 최고 설정 장기 스케줄 |
| 학교 2 | 320x240 · seed14 · 200 epoch | seed 재현성 |
| 학교 3 | 320x240 · seed15 · 200 epoch | 추가 seed 최고값 탐색 |
| 학교 4 | 480x360 최대 48 h → 320x240 최대 46 h | 중간 해상도 사전학습 |
| 학교 5 | 640x480 최대 60 h → 320x240 최대 34 h · seed11 | 고해상도 사전학습 |
| 학교 6 | 640x480 최대 60 h → 320x240 최대 34 h · seed14 | 고해상도 효과 재현 |

각 단계는 200 epoch 상한, 매 epoch checkpoint, 최대 96시간 계획이다. 고해상도 단계가 시간 제한에 도달하면 checkpoint를 보존하고 이전 단계의 `best.pt` model weight만 최종 320x240 미세조정에 전달한다. optimizer, scheduler, AMP scaler와 epoch는 새로 시작한다.

## 오늘 Raspberry Pi 객체 감지 시험

2026-08-14에 예정했던 새 모델의 Raspberry Pi/OpenCV 객체 감지 시험은 실시하지 못했다.

- 오늘 날짜의 새 detection 정확도 결과 없음
- 새 FPS·latency·temperature 측정 없음
- 새 오탐·미탐 사례 없음
- 8월 12일 임시 INT8 시험은 이전 배포 경로 기준선으로만 유지
- `results.14`는 아직 ONNX 변환·INT8 양자화·Raspberry Pi 시험 전

미실시를 실패 결과로 해석하거나 이전 임시 모델 수치를 Round 3 성능으로 연결하지 않는다.

## 보관 파일

| 파일/폴더 | 내용 |
|---|---|
| `round3_30e_summary.csv` | 실행별 최고/마지막/추세/속도/경고 집계 |
| `round3_30e_all_histories.csv` | 6개 실행의 전체 epoch history |
| `round3_30e_recomputed_summary.csv` | 공개 history에서 재계산한 핵심 비교표 |
| `analyze_round3_30e.py` | 공개된 통합 history CSV에서 순위·공통 구간 지표를 다시 계산하는 코드 |
| `configs/longrun_final/` | 6대에 필요한 9개 단계 config |
| `plans/longrun_final/` | 노트북별 96시간 실행 manifest |
| `scripts/25_run_resolution_curriculum.py` | 시간 제한·자동 복구·해상도 전환 실행기 |
| `scripts/26_smoke_test_resolution.py` | 실제 CUDA 1-batch 고해상도 사전 검사 |
| `scripts/27_summarize_longrun.py` | 최종 320x240 단계 비교 |

## 다음 순서

1. 연휴 종료 후 각 노트북의 `results/training`과 `results/longrun`을 회수한다.
2. 최종 320x240 단계의 mAP50:95, AP75, Recall, F1, tiny/small Recall과 경고를 비교한다.
3. 상위 1~2개만 FP32 ONNX와 static INT8로 변환한다.
4. grouped Valid에서 정확도 하락을 측정한다.
5. 그 뒤 Raspberry Pi 라이브 카메라에서 객체 감지·FPS·sensor-to-result·온도·오탐/미탐을 시험한다.
6. 최종 후보가 확정될 때까지 Test split은 사용하지 않는다.
