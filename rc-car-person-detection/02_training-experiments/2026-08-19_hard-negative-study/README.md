# 2026-08-19~20 Hard Negative 학습 실험

## 문제 정의

운동장 실측에서 골대·기둥·나무·가방과 같은 무인 배경을 사람으로 검출할 가능성을 줄이기 위해, 사람이 없는 실제 환경 프레임을 empty-label hard negative로 추가했다. 목표는 일반 validation 정확도를 보존하면서 고정 threshold 0.25에서 false positive(FP)를 줄이는 것이다.

## 데이터 구성

- 총 343장: train 283장, 별도 FP 평가 60장
- 출처: `picture1.zip` 113장, `picture2.zip` 111장, `picture3.zip` 119장
- 모든 샘플은 사람이 없는지 확인한 empty negative
- 원본 이미지는 라이선스·용량·중복 문제로 Git에 올리지 않음
- [`hard_negative_manifest.csv`](hard_negative_manifest.csv)에 파일명, source archive, SHA-256, group과 split을 보존

## Round 1: 단순 5 epoch 미세조정

S1, S2, R33, R46 네 source checkpoint를 같은 방식으로 5 epoch 미세조정했다. 결과는 일관되지 않았다.

- S1: FP box 13 → best 기준 21
- S2: FP box 28 → best 기준 32
- R33: FP box 2 → 3
- R46: FP box 1 → 2

즉 negative 이미지를 단순히 추가하는 것만으로는 FP가 자동 감소하지 않았다. 작은 데이터셋을 반복 노출하면 분류 calibration과 Recall이 함께 흔들릴 수 있다는 결론을 얻었다.

## Round 2: 노출량·LR·mining 분리

6대 노트북에서 다음 변수를 분리했다.

- S1: uniform x4, uniform x8, mined 80×10
- S2: uniform x4, mined 80×10
- learning rate: 1e-5 또는 2.5e-5
- 8 epoch snapshot을 모두 평가하고 사전 고정 gate로 선택

### 통과 후보

| Source | 설정 | Epoch | mAP50:95 | Recall | F1 | FP boxes | 판단 |
|---|---|---:|---:|---:|---:|---:|---|
| S1 | uniform x8, LR 2.5e-5 | 7 | 0.260041 | 0.546598 | 0.592813 | 13 → **8** | screening 후보 |
| S2 | mined 80×10, LR 2.5e-5 | 3 | 0.253740 | 0.556030 | 0.572936 | 28 → **13** | screening 후보 |

S1 epoch 7은 원본 대비 mAP가 0.000697만 감소하면서 Recall은 0.034263 증가했고 FP box는 38.5% 감소했다. S2 epoch 3도 FP를 크게 줄였지만 mAP 감소 폭이 더 컸다.

## 최종 판단

Hard-negative 후보는 FP 제어 가능성을 입증했지만 최종 배포 source로 채택하지 않았다.

- 원본 R46은 hard-negative eval에서 이미 FP box가 1개로 가장 낮았다.
- R46의 1차 hard-negative 미세조정은 FP를 개선하지 못했다.
- S1/S2 HN2 후보는 별도 ONNX/INT8 후보로 보존했지만 최종 라즈베리파이 선택은 원본 R46에서 파생한 TAIL004였다.

따라서 이 실험은 최종 모델 교체보다 **운동장 FP를 수집하고 별도 gate로 검증하는 재학습 절차**를 확립한 결과로 남긴다.

## 재현 파일

- `code/round1`, `configs/round1`: 1차 4모델 실험
- `code/round2`, `configs/round2`: 2차 6모델 exposure/mining 실험
- `results/round1`, `results/round2`: config, history, 요약
- `selected_checkpoints`: S1 epoch 7, S2 epoch 3 screening checkpoint
- [`selection_summary.csv`](selection_summary.csv): 선택 표

결과 폴더의 `AUTO CANDIDATE`는 hard-negative screening gate 통과를 뜻하며 최종 배포 채택을 뜻하지 않는다.
