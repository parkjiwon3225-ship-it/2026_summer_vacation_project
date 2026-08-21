# 2026-08-14 Round 2 일곱 결과 분석과 Round 3 최종 성능 탐색

이 폴더는 2026-08-13에 학교 노트북 6대와 집 노트북 1대에서 시작한 Round 2 결과를 비교하고, 마지막 학교 병렬 자원을 어떤 실험에 사용하기로 했는지 기록한다.

현재 전체 기준 모델은 Round 1의 `results.4`, 즉 `school2_lightweight_100e`의 epoch 37 `best.pt`다. Round 2 최고 모델이 기준을 넘지 못했으므로 최종 모델은 아직 변경하지 않는다.

## 한눈에 보는 결론

- Round 1 전체 최고: `results.4` · FPN48 exp2.0 · mAP50:95 `0.252478`
- Round 2 최고: `results.11` · FPN48 exp2.5 · mAP50:95 `0.241455`
- Round 2 학교 6개: epoch 70 완료, warning count 합계 모두 `0`
- Round 2 집 1개: `results.7 half`, epoch 50 중간 결과이며 최종 판단 보류
- 유지: FPN48, LR 0.001, box weight 2.0, radius 1.5 기준
- 비교 유지: FPN48 expansion 2.5
- 배포 보조: FPN40은 Raspberry Pi 속도·온도 이득이 클 때만 유지
- 추가 탐색 제외: FPN56, LR 0.0005, Round 2 하위 checkpoints
- Test split: 아직 사용하지 않음

## `best.pt`와 목표 수치 구분

`best.pt`는 mAP50:95가 0.25를 넘어야 저장되는 파일이 아니다. 각 실행에서 validation mAP50:95가 가장 높았던 epoch가 점수와 관계없이 `best.pt`로 저장된다.

- 최소 목표: mAP50:95 `0.25`
- 현실 목표: 약 `0.30`
- 좋은 결과로 보는 도전 목표: `0.40 이상`
- 현재 최고: `0.252478`

Confidence threshold `0.25`는 화면에 박스를 표시할 operating threshold이고, mAP50:95 `0.25`는 모델 평가 점수다. 서로 다른 값이다.

## Round 2 설정

공통 입력은 320×240, batch 8, group-aware train/valid split, AMP on이다. 안전성을 위해 AMP initial scale `1024`, growth interval `20000`을 사용했다.

| 자료 | 설정 | 계획/확보 epoch | 목적 |
|---|---|---:|---|
| `results.8` | FPN48 exp2.0 · LR 0.001 · 새 seed | 70/70 | 우승 구조 재현성 |
| `results.9` | FPN48 exp2.0 · LR 0.00075 | 70/70 | 중간 학습률 |
| `results.10` | FPN48 exp2.0 · LR 0.0005 | 70/70 | 낮은 학습률 |
| `results.13` | FPN40 exp2.0 · LR 0.001 | 70/70 | 더 작은 FPN |
| `results.12` | FPN56 exp2.0 · LR 0.001 | 70/70 | 더 큰 FPN |
| `results.11` | FPN48 exp2.5 · LR 0.001 | 70/70 | backbone capacity 증가 |
| `results.7 half` | FPN48 exp2.0 · LR 0.00075 · 새 seed | 100/50 | 집에서 장기 수렴 관찰 |

`results.9`와 `results.10`은 Windows 정책 오류 복구 과정에서 PyTorch `2.5.1+cu121` 환경을 사용했다. 나머지 주요 실행은 `2.3.1+cu121`이므로 두 결과의 미세한 차이를 학습률 효과만으로 단정하지 않는다.

## Round 2 최고 checkpoint 비교

Precision·Recall·F1은 confidence 0.25 기준이며, 크기별 Recall은 best-mAP epoch에서 읽었다.

| 순위 | 자료 | 주요 설정 | Best epoch | mAP50:95 | Precision | Recall | F1 | Tiny R | Small R |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `results.11` | FPN48 exp2.5 · LR .001 | 17 | **0.241455** | 0.721887 | 0.462673 | **0.563919** | **0.041284** | **0.385510** |
| 2 | `results.7 half` | FPN48 exp2.0 · LR .00075 | 47 | **0.224225** | 0.743479 | 0.388342 | 0.510194 | 0.004587 | 0.201728 |
| 3 | `results.13` | FPN40 exp2.0 · LR .001 | 30 | **0.218345** | 0.719966 | 0.408739 | 0.521444 | 0.015138 | 0.275839 |
| 4 | `results.9` | FPN48 exp2.0 · LR .00075 | 25 | 0.208414 | 0.788261 | 0.292325 | 0.426488 | 0.001376 | 0.133267 |
| 5 | `results.8` | FPN48 exp2.0 · LR .001 | 13 | 0.206391 | 0.849353 | 0.222267 | 0.352332 | 0.008257 | 0.160851 |
| 6 | `results.10` | FPN48 exp2.0 · LR .0005 | 21 | 0.194404 | 0.769665 | 0.318688 | 0.450741 | 0.002752 | 0.196743 |
| 7 | `results.12` | FPN56 exp2.0 · LR .001 | 16 | 0.182077 | 0.542425 | 0.422606 | 0.475077 | 0.008716 | 0.217016 |

모든 학교 실험의 최고 epoch가 30 이하였다. 집 실험을 포함해 공통 epoch 50까지만 비교해도 순위는 동일했다.

## Round 1 기준 모델과 비교

| 지표 | Round 1 `results.4` | Round 2 `results.11` | 해석 |
|---|---:|---:|---|
| mAP50:95 | **0.252478** | 0.241455 | 전체 최고는 results.4 유지 |
| AP50 | **0.553034** | 0.527424 | results.4 우세 |
| AP75 | **0.193246** | 0.189795 | 두 모델 모두 정밀 localization이 병목 |
| Precision@0.25 | 0.585655 | **0.721887** | results.11은 보수적으로 검출 |
| Recall@0.25 | **0.557562** | 0.462673 | 놓침 감소에는 results.4가 유리 |
| F1@0.25 | **0.571263** | 0.563919 | 근접하지만 results.4 우세 |
| Tiny Recall | **0.072936** | 0.041284 | results.4 우세 |
| Small Recall | **0.529412** | 0.385510 | results.4가 크게 우세 |

## 목표 점수에 미달한 이유

1. 약 32만 파라미터의 초경량 커스텀 모델을 pretrained backbone 없이 처음부터 학습한다.
2. 320×240 letterbox에서 Valid 박스의 약 41.7%가 32 px 미만이고 약 17.5%가 16 px 미만이다.
3. 전체 최고도 AP50 `0.5530`에 비해 AP75 `0.1932`가 낮아 사람의 존재보다 박스 위치 정밀도가 병목이다.
4. Round 2 학교 실행은 70-epoch cosine schedule이라 100-epoch 기준보다 학습률이 빠르게 감소했다.
5. 같은 FPN48·LR 0.001 계열에서도 seed와 schedule에 따라 `0.2525`와 `0.2064`가 나와 재현성 확인이 필요하다.

Round 2는 정확도 향상에는 성공하지 못했지만, FPN56과 낮은 LR을 제거하고 FPN48로 탐색 범위를 줄였다는 의미가 있다. 같은 변수를 반복해서 넓게 탐색하는 대신 재현성·localization·small-person assignment를 직접 검증하는 단계로 이동한다.

## 현재 후보

| 역할 | checkpoint | 판단 |
|---|---|---|
| 정확도 기준 A | Round 1 `results.4 best.pt` | mAP·Recall·small recall 전체 최고 |
| capacity 비교 B | Round 2 `results.11 best.pt` | Round 2 최고지만 더 크고 작은 사람 성능은 낮음 |
| 배포 비교 C | Round 2 `results.13 best.pt` | FPN40 속도·온도 이득 확인용이며 정확도 최종 후보는 아님 |
| 결과 대기 | `results.7 half` | 집에서 100 epoch 진행 중 |

## Round 3: 마지막 학교 6대 성능 탐색

학교에서 약 6시간 사용할 수 있어 약 39~44 epoch가 예상된다. 모든 config의 `epochs`는 100으로 둔다. 학습을 약 40 epoch에서 중단해도 cosine scheduler는 100-epoch 기준으로 천천히 감소하고, 완료된 매 epoch마다 `best.pt`, `last.pt`, `history.csv`가 저장된다.

| 노트북 | 설정 | 검증하려는 가설 |
|---:|---|---|
| 1 | FPN48 exp2.0 · seed `20260811` | Round 1 results.4를 안전 AMP에서 정확 재현 |
| 2 | FPN48 exp2.0 · seed `20260814` | 최고 구조의 seed 재현성 |
| 3 | FPN48 exp2.5 · seed `20260811` | 100-epoch schedule에서 capacity 효과 |
| 4 | FPN48 exp2.5 · seed `20260814` | 확장 모델의 seed 재현성 |
| 5 | FPN48 exp2.0 · box weight `2.5` | AP75와 박스 위치 정밀도 개선 |
| 6 | FPN48 exp2.0 · center radius `2.0` | positive target 확대와 small-person Recall 개선 |

공통 설정:

- 입력 320×240
- batch 8
- LR 0.001
- FPN48
- 100 planned epochs
- AMP initial scale 1024 / growth interval 20000
- checkpoint와 metrics 매 epoch
- Test split 미사용

정확한 config는 [`configs/round3`](configs/round3/)에 있다.

## 부분 결과 판단 기준

- mAP50:95 `0.245 이상`: 강한 최종 후보
- mAP50:95 `0.235 이상`이면서 상승 중: 집에서 계속 학습할 후보
- Small 16~32 px Recall `0.40 이상`: 작은 사람 개선 후보
- F1@0.25 `0.56 이상`: 현재 최고 수준
- mAP가 비슷하더라도 AP75 또는 small recall이 유의하게 증가하면 보조 후보로 유지
- 최종 비교에는 `last.pt`가 아니라 `best.pt` 사용

## 17시 결과 회수

1. 가능하면 epoch 결과가 출력된 직후 `Ctrl+C`를 한 번 누른다.
2. 마지막 완료 epoch가 저장됐다는 메시지와 프롬프트 복귀를 확인한다.
3. `summarize_round3.py`로 현재 best를 확인한다.
4. `collect_round3_results.py`로 config, device, history, logs, `best.pt`, `last.pt`를 ZIP으로 모은다.
5. 각 노트북 ZIP을 USB로 회수한다.

## 포함 파일

| 위치 | 내용 |
|---|---|
| `round2_full_summary.csv` | 학교 완료 결과와 집 중간 결과의 best·last·안정성 요약 |
| `round2_best_through_epoch50.csv` | 모든 실행을 epoch 50까지만 사용한 공정 비교 |
| `round2_epoch50_snapshot.csv` | 정확히 epoch 50의 지표 |
| `round1_reference_summary.csv` | Round 1 기준 모델 비교 자료 |
| `round2_all_histories.csv` | Round 2 전체 학습 곡선 원자료 |
| `configs/round3/` | 6대 최종 병렬 실험 config |
| `scripts/analyze_round2_archives.py` | 결과 ZIP 재분석 스크립트 |
| `scripts/summarize_round3.py` | 진행 중인 Round 3 best 요약 |
| `scripts/collect_round3_results.py` | 부분 결과를 USB용 ZIP으로 수집 |

Round 2 ZIP을 다시 분석하려면 프로젝트 환경에 `pandas`, `numpy`가 설치된 상태에서 다음처럼 실행한다.

```bash
python scripts/analyze_round2_archives.py --source PATH_TO_RESULT_ZIPS --output round2_analysis
```

## 주의

- `results.7 half`는 최종 학습 결과가 아니다. 완료 결과가 확보되면 이 문서와 CSV를 갱신한다.
- `results.9`, `results.10`의 환경 차이 때문에 낮은 LR의 미세 차이를 단독 효과로 단정하지 않는다.
- 최종 후보 1~2개가 확정되기 전까지 Test split은 사용하지 않는다.
- 운동장 hard-negative 추가 학습과 최종 FP32/INT8 선택은 Round 3 이후 단계다.
