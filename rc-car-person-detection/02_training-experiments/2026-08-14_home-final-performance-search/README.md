# results.14 기반 홈 최종 성능 탐색

기준일: 2026-08-14

데이터: `data/processed/v1_grouped`

평가 split: Valid

Test split: 사용하지 않음

## 목적

학교 6대의 연휴 장시간 학습과 별도로, 집의 MX570 A 4GB 대여 노트북에서 월요일까지 가장 좋은 320×240 후보를 확보한다. Round 3에서 검증된 FPN48 구조를 다시 바꾸지 않고 충분히 연속 학습한 뒤 두 개의 낮은 learning rate로 미세조정한다.

## 출발점 선택

현재 전역 최고는 `results.14`의 epoch 29 `best.pt`다.

| 자료 | 상태 | best epoch | mAP50:95 | AP50 | AP75 | Precision | Recall | F1 | tiny R | small R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| results.14 | Round 3 seed11 | 29 | **0.253238** | 0.550494 | 0.201683 | 0.6732 | **0.5103** | **0.5805** | **0.0739** | **0.4533** |
| results.7 | 집 LR 0.00075 | 62 | 0.224346 | 0.496195 | 0.174479 | **0.7556** | 0.3792 | 0.5049 | 0.0073 | 0.2173 |

`results.7`은 epoch 87까지 진행됐다. epoch 62 이후 25 epoch 동안 best mAP를 갱신하지 못했고, 마지막 mAP는 0.208602, F1은 0.491211, small recall은 0.166168, Valid loss는 1.647612였다. warning은 0건이므로 오류가 아니라 plateau와 과적합 경향으로 해석한다. 남은 13 epoch를 계속하기보다 더 좋은 `results.14`에서 최종 탐색을 시작한다.

## 유지한 모델 구조

```text
Input 320×240
  → DSConv + Residual backbone · expansion 2.0
  → P2/P3/P4/P5 Lightweight FPN · channels 48
  → Anchor-free head
  → box weight 2.0 · quality weight 1.0 · center radius 1.5
```

Round 3에서 expansion 2.5, box weight 2.5, radius 2.0이 더 낮았기 때문에 남은 일정에는 구조를 다시 탐색하지 않는다. 모델 구조를 고정하고 학습 완성도와 localization을 높이는 편이 현재 증거와 일정에 맞다.

## 3단계 자동 탐색

| 단계 | 출발점 | 방식 | LR | 범위 | 시간 상한 | 목적 |
|---:|---|---|---:|---:|---:|---|
| 1 | results.14 `last.pt` epoch 30 | exact resume | 0.001 | epoch 100까지 | 24 h | 기존 optimizer·scheduler를 유지해 충분히 수렴 |
| 2 | 1단계까지의 전역 최고 `best.pt` | weights-only init | 0.00025 | 40 epoch | 14 h | 최고 checkpoint 주변 미세조정 |
| 3 | 2단계까지의 전역 최고 `best.pt` | weights-only init | 0.00010 | 24 epoch | 8 h | 작은 LR로 최종 정밀조정 |

총 GPU 시간 상한은 46시간이다. 각 단계가 끝나면 직전 단계만 채택하지 않고 `results.14` 원본을 포함한 모든 후보의 Valid mAP50:95를 다시 비교한다.

## resume 설계에서 확인한 함정

- `--resume`은 model뿐 아니라 optimizer, scheduler, AMP scaler와 epoch를 복구한다. config에서 LR만 바꿔도 실제로는 이전 optimizer LR이 복원될 수 있다.
- results.14 checkpoint의 cosine scheduler는 `T_max=100`이다. 이를 epoch 200으로 그대로 resume하면 epoch 100 이후 LR이 다시 상승할 위험이 있다.
- 따라서 1단계만 동일 설정으로 epoch 100까지 resume하고, LR을 바꾸는 2·3단계는 `--init-weights`를 사용한다.
- 외부 `last.pt`를 새 experiment 폴더로 resume하면 기존 best score는 복원돼도 이를 넘지 못할 경우 새 `best.pt`가 생성되지 않을 수 있다. 실행기는 원본 best checkpoint를 먼저 별도 보존한다.
- checkpoint epoch보다 뒤의 중복 history 행을 정리하고 config가 checkpoint 구조와 일치하는지 확인한다.

## 보존 및 무결성

| 파일 | SHA-256 |
|---|---|
| `reference/results14_seed11/checkpoints/best.pt` | `D44DB15BCB623EDE678C48150CE7EF6965F871FA0D3DF80D8BEC2E900F8FF27A` |
| `reference/results14_seed11/checkpoints/last.pt` | `1E4633F06FE33E91F069DDF1E8BF9EBFE2487A75561EBC764BD854345A9B12FA` |
| 사용자 전달 ZIP | `90B6340DAE4CB71BC7FFA24DAF30652037EB7BA2F94CC4FD6B925BCE6163D536` |

원본 seed checkpoint는 읽기 전용 출발점으로 취급한다. 새 결과는 별도 `results/training/home_final_*` 폴더에 기록한다.

## 실행

기존 프로젝트 root에서 환경과 dataset을 검증한다.

```powershell
conda activate rc-person-detector
python -m pip install -e .
python verify_home_search.py --root . --runtime
```

`STATUS : PASS` 확인 후 실행한다.

```powershell
python scripts/28_run_home_final_search.py --root . --plan plans/home_final_search.json
```

중단되면 같은 명령을 다시 실행한다. 요약만 다시 만들 때는 다음을 사용한다.

```powershell
python scripts/29_summarize_home_final_search.py --root .
```

## 결과 위치

```text
results/training/home_final_s1_continue_results14_to100/
results/training/home_final_s2_finetune_lr0250_40e/
results/training/home_final_s3_polish_lr0100_24e/
results/home_final_search/home_final_search_v1/home_final_ranking.csv
results/home_final_search/home_final_search_v1/final_candidate/best.pt
results/home_final_search/home_final_search_v1/final_candidate/selection.json
```

## 다음 판단

1. 학교 장시간 6개와 집 최종 후보를 동일 grouped Valid 기준으로 비교한다.
2. mAP50:95뿐 아니라 AP75, Recall, F1, tiny/small recall, Valid loss와 경고를 함께 본다.
3. 상위 1~2개만 FP32 ONNX와 static INT8 QDQ 후보로 변환한다.
4. FP32 대비 INT8 정확도 하락을 grouped Valid에서 먼저 확인한다.
5. Raspberry Pi에서 FPS, inference latency, sensor-to-result, 온도와 오탐·미탐을 측정한다.
6. 최종 후보 확정 전까지 Test split은 사용하지 않는다.

## 파일 구성

- `configs/`: 세 단계 학습 설정
- `plans/home_final_search.json`: 단계 순서와 시간 상한
- `scripts/28_run_home_final_search.py`: 전역 최고 선택·재개·시간 제한 실행기
- `scripts/29_summarize_home_final_search.py`: 후보 비교표 생성
- `reference/results14_seed11/`: 출발 checkpoint, config, device, history와 provenance
- `reference/results7_home_87e/`: 중단 판단의 근거가 된 config, device, history와 runner log

데이터셋과 TensorBoard 임시 파일, 전체 결과 ZIP은 저장소에 포함하지 않는다.
