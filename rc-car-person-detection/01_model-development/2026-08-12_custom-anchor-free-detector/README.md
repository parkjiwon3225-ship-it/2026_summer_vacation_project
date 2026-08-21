# RC-Car Custom Person Detector

RC카의 제한된 연산 자원에서 실시간 사람 감지를 수행하기 위해 처음부터 설계한 경량 객체 탐지 모델입니다. YOLO 모델을 사용하는 프로젝트가 아니며, 데이터 라벨 저장 형식만 YOLO 형식(`class cx cy w h`)을 사용합니다.

> 캘리브레이션 팀과 라이브 영상 연동을 위한 입력·출력 규약과 확인 사항은 [`04_team-integration/2026-08-12_calibration-interface-guide`](../../04_team-integration/2026-08-12_calibration-interface-guide/)를 참고하세요.

## 현재 상태

- Group-aware 데이터 분할 및 무결성 검사: 완료
- 커스텀 모델·손실·추론·평가 구현: 완료
- CPU/CUDA 단위 테스트와 mini-overfit: 통과
- NVIDIA GeForce MX570 A(4GB) 전체 데이터 2-epoch pilot: 통과
- 100-epoch 기준 모델 및 5개 비교 실험: 1차 분석 완료
- 1차 종합 1위: FPN48 `results.4`, best mAP50:95 `0.2525`
- 2차 병렬 실험: 2026-08-13 시작, 결과 회수 예정
- Test split 최종 평가는 모델 선택이 끝날 때까지 보류

이 폴더에는 소스 코드와 재현 설정만 포함합니다. 데이터셋, 학습 결과, 체크포인트와 개인 PC 경로는 포함하지 않습니다. 1차 비교 결과는 [`02_training-experiments`](../../02_training-experiments/2026-08-13_round1-six-laptop-study/)에, 현재 최고 checkpoint 전달본은 [`04_team-integration`](../../04_team-integration/2026-08-13_results4-training-handoff/)에 있습니다.

## 모델 구조

```text
RGB 320x240
  -> DSConv + Residual 경량 Backbone
  -> P2/P3/P4/P5 Lightweight FPN
  -> Anchor-free Detection Head
  -> Classification + Quality + LTRB Box Regression
  -> Decode + Pure PyTorch NMS
```

- 약 345K parameters(기준 모델)
- FCOS 계열 center sampling 및 scale별 regression range
- Focal classification loss
- BCE quality loss
- Quality-weighted GIoU box loss
- P2 feature level을 사용해 작은 사람 탐지 강화
- AMP, gradient clipping, atomic checkpoint, resume 지원

자세한 설계는 [`docs/model_design_v1.md`](docs/model_design_v1.md)를 참고하세요.

## 데이터 분할

동일 원본 이미지 변형이나 같은 영상 프레임이 서로 다른 split에 들어가는 누수를 막기 위해 source group 단위로 분할했습니다.

| Split | Images | Boxes |
|---|---:|---:|
| Train | 12,322 | 88,750 |
| Valid | 1,531 | 12,404 |
| Test | 1,531 | 12,306 |
| Total | 15,384 | 113,460 |

- Group leakage: `0`
- Missing/orphan/empty labels: `0`
- 무결성 검사: `PASS`

저장소에는 데이터셋을 올리지 않습니다. 로컬에서 다음 구조로 준비해야 합니다.

```text
data/processed/v1_grouped/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

## 환경 설치

Anaconda Prompt에서 이 폴더로 이동한 다음 실행합니다.

```powershell
conda env create -f environment-school.yml
conda activate rc-person-detector
python -m pip install -e .
```

이미 환경을 만들었다면 다음 두 줄만 실행합니다.

```powershell
conda activate rc-person-detector
python -m pip install -e .
```

## 사전 검증

Windows에서는 다음 파일을 실행합니다.

```powershell
00_verify_school_setup.bat
```

개별 검사는 다음 순서로 실행할 수 있습니다.

```powershell
python scripts/13_test_training_step.py
python scripts/14_test_inference.py
python scripts/17_test_metrics.py
python scripts/18_test_monitoring.py
python scripts/19_validate_experiments.py
python scripts/15_mini_overfit.py --device cuda
```

모든 항목이 `PASS`인 것을 확인한 뒤 전체 학습을 시작합니다.

## 학습

기준 모델 100 epoch:

```powershell
python scripts/16_train.py --config configs/experiments/home_baseline_100e.json
```

학습을 중단했다가 재개할 때:

```powershell
python scripts/16_train.py --config configs/experiments/home_baseline_100e.json --resume results/training/home_baseline_100e/checkpoints/last.pt
```

2026-08-14 장시간 학습 코드에서는 다음 기능을 추가했습니다.

- config의 `image_width`, `image_height`를 실제 학습·평가·decode에 적용
- `--init-weights`로 optimizer/scheduler 상태 없이 model weight만 전달
- `--max-runtime-hours`로 epoch 경계에서 안전 종료
- `training_status.json`에 종료 원인·마지막 epoch·최고 mAP 기록
- 480/640 사전학습 후 최종 320×240 미세조정 자동 전환
- 비정상 종료 후 `last.pt` 자동 재개

노트북별 최종 계획은 `plans/longrun_final`, 단계별 설정은 `configs/longrun_final`에 있습니다.

집 노트북 최종 성능 탐색에는 다음 안전장치를 추가했습니다.

- `results.14`의 last checkpoint는 동일 config로 epoch 100까지만 정확히 resume
- learning rate를 바꾸는 다음 단계는 `--init-weights`로 model weight만 전달
- 각 단계 완료 후 이전 모든 후보를 다시 비교해 전역 최고 checkpoint 선택
- checkpoint epoch 이후의 중복 history를 정리하고 config 일치 여부 검사
- 단계별 시간 상한과 자동 재시도, 같은 명령으로 중단 복구

실행기는 `scripts/28_run_home_final_search.py`, 결과 요약은 `scripts/29_summarize_home_final_search.py`, 설정은 `configs/home_final`, 계획은 `plans/home_final_search.json`에 있습니다.

GitHub checkout의 `home_final_search.json`은 중복 checkpoint를 만들지 않도록 `02_training-experiments/2026-08-14_home-final-performance-search/reference/results14_seed11`을 출발점으로 참조합니다. USB 전달용 독립 패키지는 같은 파일을 자체 `seeds/results14_seed11`에 포함합니다.

학교 GPU 5대의 실험 배정과 정확한 명령은 [`01_노트북별_실험배정.txt`](01_노트북별_실험배정.txt)에 정리했습니다.

## 6개 실험의 목적

모든 실험은 같은 데이터, seed, 입력 크기, batch와 epoch를 사용합니다. 한 번에 한 가지 핵심 요소만 변경해 성능 변화의 원인을 해석할 수 있도록 구성했습니다.

| 설정 | 변경점 | 목적 |
|---|---|---|
| home baseline | 기준값 | 비교 기준 |
| school1 | learning rate `0.0005` | 더 안정적인 수렴 확인 |
| school2 | FPN channels `48` | 경량화 가능성 확인 |
| school3 | FPN `80`, expansion `2.5` | 모델 용량 부족 여부 확인 |
| school4 | box loss weight `3.0` | 박스 위치 정확도 강화 |
| school5 | center radius `2.0` | 작은 사람 recall 강화 |

## 기록되는 지표

각 epoch마다 다음 정보를 `history.csv`, TensorBoard와 로그에 저장합니다.

- Train/Valid total, classification, quality, box loss
- mAP50:95, AP50, AP75
- Precision, Recall, F1, 최적 F1 threshold
- TP, FP, FN, detection accuracy, mean IoU
- Tiny(`<16px`) 및 Small(`16~32px`) recall
- Gradient norm, learning rate, AMP scale
- Peak VRAM, 처리 속도, epoch 시간
- NaN/Inf 및 경고 횟수

`best.pt`는 Valid mAP50:95 기준으로 선택합니다.

## 확인된 pilot 결과

MX570 A 4GB, 전체 Train/Valid 데이터, AMP, batch 4 기준입니다.

| Epoch | Train loss | Valid loss | mAP50:95 | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.2378 | 1.9952 | 0.0454 | 0.5537 | 0.0752 | 0.1324 |
| 2 | 1.9061 | 1.8867 | 0.0820 | 0.5861 | 0.1674 | 0.2604 |

초기 AMP overflow를 확인한 뒤 target dtype을 FP32로 고정하고 GradScaler 초기 scale과 growth interval을 안정화했습니다. 최종 성능은 100-epoch 실험과 독립적인 최종 Test 평가가 끝난 뒤 기록할 예정입니다.

## 저장소 제외 항목

- `archive/`, `data/`: 원본 및 가공 데이터
- `results/`, `logs/`: 실행 결과
- `checkpoints/`, `*.pt`, `*.onnx`: 모델 파일
- `dist/`, `*.zip`: 전달용 묶음
- Python/Jupyter 캐시

데이터와 대용량 산출물은 Git 또는 별도 스토리지 정책이 정해진 뒤 관리합니다.

## 실험 기록

- [2026-08-13: 6대 노트북 1차 병렬 학습 분석 및 2차 실험 설계](../../02_training-experiments/2026-08-13_round1-six-laptop-study/README.md)
  - 공통 48 epoch 및 최고 checkpoint 비교
  - 중단 원인과 AMP gradient overflow 해석
  - FPN48 중심의 2차 6대 병렬 실험 설계와 시작 상태
- [2026-08-14: Round 2 7개 분석과 Round 3 설정](../../02_training-experiments/2026-08-14_round2-seven-run-analysis-and-round3-search/README.md)
- [2026-08-14: Round 3 30-epoch 결과와 연휴 장시간 6대 계획](../../02_training-experiments/2026-08-14_round3-30epoch-screening-and-longrun-plan/README.md)
- [2026-08-14: results.7 장기 추세와 results.14 기반 홈 최종 성능 탐색](../../02_training-experiments/2026-08-14_home-final-performance-search/README.md)
