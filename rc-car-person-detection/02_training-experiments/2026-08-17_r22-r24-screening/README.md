# 2026-08-17 R22-R24 실험 정리

기준일: 2026-08-17

데이터: `data/processed/v1_grouped`

평가 split: Valid

Test split: 사용하지 않음

## 오늘 실험 범위

오늘 새로 확인한 실험은 정확히 3개다.

1. `results.22` — R22: weight decay 0.0002
2. `results.23` — R23: direct 384x288
3. `results.24` — PC방 direct 480x360

기존 320x240 최고 모델은 오늘 실험이 아니라 비교 기준으로만 사용한다.

비교 기준:
- 320x240
- FPN channels 48
- Backbone expansion 2.0
- Box loss weight 2.0
- Quality loss weight 1.0
- Center sampling radius 1.5
- best epoch 35
- validation mAP50:95 **0.260738**

## results.22 — R22 higher weight decay

실험명: `r22_fpn48_wd200_seed11_100e`

변경점:
- weight decay `0.0001 -> 0.0002`
- 나머지 구조와 학습 조건은 기준 계열 유지

결과:
- best epoch: **18**
- best mAP50:95: **0.215903**
- Precision: 0.756974
- Recall: 0.350048
- F1: 0.478721
- tiny recall: 0.005046
- small recall: 0.216351
- last epoch: 51
- last mAP50:95: 0.201106

판단:
- 기준 최고 0.260738보다 크게 낮다.
- 후반부 성능 개선 없이 하락/plateau 경향을 보여 중단했다.
- weight decay 0.0002는 현재 주력 후보에서 제외한다.

## results.23 — R23 direct 384x288

실험명: `r23_fpn48_384x288_seed11_100e`

변경점:
- input `320x240 -> 384x288`
- 모델 구조와 나머지 주요 하이퍼파라미터는 기준 계열 유지

결과:
- best epoch: **16**
- best mAP50:95: **0.206360**
- Precision: 0.873427
- Recall: 0.190261
- F1: 0.312459
- tiny recall: 0.000000
- small recall: 0.020499
- last epoch: 24
- last mAP50:95: 0.170105

판단:
- Precision은 높았지만 Recall과 small-person recall이 크게 무너졌다.
- epoch 16 이후 mAP가 하락했다.
- direct 384x288 경로는 중단한다.

## results.24 — PC방 direct 480x360

실험명: `pcroom_480x360_direct_seed11_100e`

실행 환경:
- NVIDIA GeForce RTX 5060 Ti 8GB
- PyTorch 2.7.1+cu128
- batch 8
- AMP on

변경점:
- input `320x240 -> 480x360`
- fresh training
- FPN48 / expansion2.0 등 기준 구조 유지

mAP 진행:
- E1: 0.019940
- E2: 0.063316
- E5: 0.132589
- E10: 0.187123
- E15: 0.208902
- E20: 0.219496
- E25: **0.224773**

E25 주요 지표:
- mAP50:95: **0.224773**
- Precision: 0.674725
- Recall: 0.450016
- F1: 0.539924
- tiny recall: 0.015480
- small recall: 0.233048

5-epoch mAP 증가폭:
- E5 -> E10: +0.054534
- E10 -> E15: +0.021779
- E15 -> E20: +0.010594
- E20 -> E25: +0.005277

판단:
- mAP는 계속 올랐지만 개선폭이 빠르게 감소했다.
- 기준 최고 0.260738과 차이가 남아 있고, 최근 증가폭으로는 추격 기대값이 낮아 E25에서 중단했다.
- direct 480x360도 현재 주력 후보에서 제외한다.

### PC방 속도 조사

초기:
- `num_workers=2`
- `metrics_every=1`
- E1 약 410 s
- E2 약 422 s

GPU 사용률이 낮아 worker 병목 가능성을 확인하기 위해 `num_workers=8` speed test를 실행했지만:
- 1 epoch 약 488 s

오히려 느려져 worker 증가는 폐기했다.

학습 코드에서 validation loss 평가와 detection/NMS/mAP 평가가 별도 pass로 수행되어 metric epoch가 특히 느렸다.
그래서 본 실험은 E3부터 `metrics_every=5`로 변경했다.

대략:
- 일반 epoch: 270~295 s
- metric epoch: 467~480 s

주의:
- `metrics_every=5` 전환 이후 원본 `history.csv`의 비측정 epoch metric column 정렬에 문제가 있다.
- GitHub에는 실제 mAP가 계산된 epoch만 남긴 `histories/pcroom_480x360_metric_epochs_clean.csv`를 보관한다.
- checkpoint 자체는 정상이며 E25 best mAP 0.224773이 보존됐다.

## 종합 비교

| 구분 | 실험 | 최고 mAP50:95 | 판단 |
|---|---|---:|---|
| Reference | 기존 320x240 champion | **0.260738** | 유지 |
| results.22 | WD 0.0002 | 0.215903 | 제외 |
| results.23 | direct 384x288 | 0.206360 | 제외 |
| results.24 | direct 480x360 | 0.224773 | 제외 |

오늘 결과만 보면 단순 direct high-resolution은 기준 320x240보다 우세하지 않았다.

단, 학교에서 진행 중인 `고해상도 pretraining -> 320x240 fine-tuning`은 학습 경로가 다르므로 별도 실험으로 판단한다.

## 다음 실험

집 노트북 다음 실험은 R25:
- Input 320x240
- FPN48
- expansion 2.0
- box weight 2.0
- center radius 1.5
- quality loss weight `1.0 -> 1.5`
- seed 20260811
- fresh training

R25는 오늘 R22-R24의 결과가 아니라 다음 탐색 실험으로 분리한다.
