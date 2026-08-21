# 02. 학습 실험

날짜별 학습 설정, 수치, 비교 결과와 선택·중단 근거를 보관합니다. 단순 checkpoint 모음이 아니라 다음 실험을 결정한 이유와 최종 모델이 만들어진 경로를 확인하는 영역입니다.

## 실험 이력

| 날짜 | 실험 | 핵심 결론 |
|---|---|---|
| 2026-08-13 | [`round1-six-laptop-study`](2026-08-13_round1-six-laptop-study/) | 6대 병렬 비교에서 FPN48 results.4가 mAP·F1·small recall 종합 우위 |
| 2026-08-14 | [`round2-seven-run-analysis-and-round3-search`](2026-08-14_round2-seven-run-analysis-and-round3-search/) | 7개 결과를 비교해 FPN48 중심으로 축소하고 seed·capacity·box 정밀도 변수를 분리 |
| 2026-08-14 | [`round3-30epoch-screening-and-longrun-plan`](2026-08-14_round3-30epoch-screening-and-longrun-plan/) | 짧은 screening으로 FPN48/exp2.0/box2.0/radius1.5를 기준화하고 장시간 실험 구성 |
| 2026-08-14 | [`home-final-performance-search`](2026-08-14_home-final-performance-search/) | resume와 단계별 미세조정으로 집 노트북 장기 학습 자동화 |
| 2026-08-17 | [`r22-r24-screening`](2026-08-17_r22-r24-screening/) | weight decay·384 직접 입력·480 직접 입력이 기존 champion을 넘지 못해 중단 |
| 2026-08-18 | [`r25-r31-and-longrun-study`](2026-08-18_r25-r31-and-longrun-study/) | 640×480 계열에서 mAP 개선을 확인하고 selective INT8 후보를 생성 |
| 2026-08-18 | [`R32~R37 resolution screening`](2026-08-20_final-resolution-search/) | 512~768 해상도 병렬 탐색. 최종 비교 폴더에 R32~R50 집계 결과를 통합 |
| 2026-08-19 | [`hard-negative-study`](2026-08-19_hard-negative-study/) | 운동장 empty-negative 343장 실험. S1·S2 FP는 감소했지만 R46 개선이 일관되지 않아 최종 원본에는 미적용 |
| 2026-08-20 | [`final-resolution-search`](2026-08-20_final-resolution-search/) | R33의 mAP 최고와 R46의 배포 균형을 비교해 `R46 / 448×336 / seed15 / epoch25`를 최종 경량화 원본으로 선정 |

## 최종 후보 비교

| run | 입력 | mAP50:95 | Precision | Recall | F1 | small recall | 판단 |
|---|---:|---:|---:|---:|---:|---:|---|
| R33 | 576×432 | **0.301354** | 0.884414 | 0.394792 | 0.545900 | 0.076220 | accuracy champion |
| R36 | 704×528 | 0.298928 | 0.837749 | **0.430829** | **0.569025** | 0.031490 | 정확하지만 Pi 비용 큼 |
| R45 | 448×336 | 0.248999 | 0.806294 | 0.362706 | 0.500334 | 0.107104 | 같은 크기의 seed11 비교군 |
| **R46** | **448×336** | 0.261832 | 0.817398 | 0.412851 | 0.548610 | **0.155005** | **최종 배포 원본** |
| R49 | 576×432 | 0.286423 | **0.919874** | 0.235085 | 0.374470 | 0.017590 | precision 편향으로 제외 |

R46은 최고 mAP 모델이 아닙니다. 하지만 320×240 카메라와 같은 4:3 비율을 유지하면서 R33보다 입력 픽셀이 약 40% 적고, Recall과 small-person Recall이 더 높았습니다. 놓침을 중요하게 보는 RC카 프로젝트와 Raspberry Pi 계산량을 함께 고려한 선택입니다.

## 보존 규칙

- 실험별 README, 실행 config, history, summary를 우선 보존합니다.
- 최종 선택에 직접 사용된 대표 `best.pt`만 보존합니다.
- 원본 학습 데이터, 중복 checkpoint, cache, 로컬 절대경로는 제외합니다.
- 결과 비교에는 validation을 사용하고 test split은 최종 모델의 마지막 평가 전까지 탐색에 사용하지 않습니다.
