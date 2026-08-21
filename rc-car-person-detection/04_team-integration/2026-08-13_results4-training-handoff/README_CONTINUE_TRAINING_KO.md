# Results.4 이어학습 참고자료

작성일: 2026-08-13
대상: 캘리브레이션 팀
목적: AI 팀의 `results.4` FPN48 모델을 동일한 조건에서 안전하게 이어서 학습하고 결과를 다시 비교하기

## 1. 먼저 알아야 할 내용

이 모델은 완성된 최종 모델이 아니라 1차 병렬 실험에서 가장 좋은 결과를 기록한 후보 모델입니다.

- 구조: DSConv + Residual backbone + 경량 FPN + anchor-free detection head
- 입력: RGB `320×240` letterbox
- 클래스: `person` 1개
- 파라미터: 323,546개
- 최고 validation mAP50:95: `0.252478`
- 최고 checkpoint epoch: 37
- 마지막 완료 epoch: 48
- Test split: 아직 사용하지 않음

`best`는 F1이 아니라 validation mAP50:95 기준입니다.

## 2. 어떤 checkpoint를 사용해야 하는가

### 기존 학습을 그대로 계속할 때 — 권장

```text
model/results4_fpn48_last.pt
```

- epoch 48 저장본
- epoch 49부터 이어서 학습
- 모델 가중치, optimizer, scheduler, AMP GradScaler 상태를 모두 복구
- 기존 results.4 실행의 정확한 연속 실험

### 최고 성능 지점에서 새 분기를 만들 때

```text
model/results4_fpn48_best.pt
```

- epoch 37 저장본
- epoch 38부터 다시 학습
- 기존 38~48 epoch와는 다른 학습 경로가 만들어질 수 있음
- 실험 이름에 `branch_from_best`라고 표시할 것

단순히 학습을 더 이어보는 목적이면 반드시 `last.pt`를 사용합니다.

## 3. 필요한 파일

GitHub 전달 폴더에는 다음 항목이 포함돼 있습니다.

```text
2026-08-13_results4-training-handoff/
├─ model/
│  ├─ results4_fpn48_best.pt
│  └─ results4_fpn48_last.pt
├─ results/
│  ├─ config.json
│  ├─ device.json
│  ├─ history.csv
│  └─ warnings.log
└─ training_code/
   ├─ environment-school.yml
   ├─ pyproject.toml
   ├─ configs/school2_lightweight_100e.json
   ├─ scripts/16_train.py
   ├─ scripts/resume_results4.py
   └─ src/rc_detector/*.py
```

데이터셋은 GitHub에 포함되지 않습니다. AI 팀이 사용한 다음 폴더를 USB 등으로 별도 전달받아야 합니다.

```text
data/processed/v1_grouped/
├─ train/images
├─ train/labels
├─ valid/images
├─ valid/labels
├─ test/images
└─ test/labels
```

이어학습 과정에서는 `train`과 `valid`만 사용합니다. `test`는 최종 후보가 결정될 때까지 열거나 반복 평가하지 않습니다.

## 4. 환경 준비

Anaconda Prompt를 열고 환경을 활성화합니다.

```powershell
conda activate rc-person-detector
```

확인:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

정상 예:

```text
cuda: True
gpu: NVIDIA GeForce MX570 A
```

전달 폴더의 학습 코드로 이동하고 설치합니다.

```powershell
cd /d C:\경로\2026-08-13_results4-training-handoff\training_code
python -m pip install -e .
```

`pip install -e .`는 반드시 `pyproject.toml`이 있는 `training_code` 폴더에서 실행합니다.

## 5. 데이터 경로 확인

예를 들어 원본 프로젝트가 다음 위치라면:

```text
C:\Users\USER\Desktop\RC\RC
```

아래 폴더들이 실제로 존재해야 합니다.

```text
C:\Users\USER\Desktop\RC\RC\data\processed\v1_grouped\train\images
C:\Users\USER\Desktop\RC\RC\data\processed\v1_grouped\train\labels
C:\Users\USER\Desktop\RC\RC\data\processed\v1_grouped\valid\images
C:\Users\USER\Desktop\RC\RC\data\processed\v1_grouped\valid\labels
```

`--data-root`에는 `data` 폴더 자체가 아니라 `data`를 포함하는 프로젝트 루트를 입력합니다.

## 6. 권장 실행 명령

전달 폴더의 `training_code` 위치에서 실행합니다.

```powershell
python scripts\resume_results4.py --data-root "C:\Users\USER\Desktop\RC\RC" --checkpoint last
```

정상이라면 다음과 같이 표시돼야 합니다.

```text
start_epoch : 49
```

메모리가 부족한 경우에만 batch 크기를 낮춥니다.

```powershell
python scripts\resume_results4.py --data-root "C:\Users\USER\Desktop\RC\RC" --checkpoint last --batch-size 4
```

Batch 크기를 바꾸면 기존 실험과 완전히 동일한 조건은 아니므로 결과 기록에 반드시 남깁니다.

## 7. 변경하지 말아야 할 기준값

동일 조건 이어학습에서는 다음 값을 변경하지 않습니다.

| 항목 | 값 |
|---|---:|
| 입력 크기 | 320×240 |
| FPN channels | 48 |
| Backbone expansion | 2.0 |
| Learning rate 초기 설정 | 0.001 |
| Weight decay | 0.0001 |
| Box loss weight | 2.0 |
| Quality loss weight | 1.0 |
| Center sampling radius | 1.5 |
| Score threshold | 0.25 |
| NMS IoU threshold | 0.5 |
| Seed | 20260811 |

checkpoint를 resume하면 optimizer와 scheduler 상태가 복원되므로, epoch 49의 실제 학습률은 초기값 `0.001`보다 낮을 수 있습니다. 이것은 정상입니다.

다른 값을 시험하려면 기존 이어학습과 섞지 말고 새로운 실험 이름과 별도 결과 폴더를 사용합니다.

## 8. 학습 중 확인할 지표

매 epoch 다음 값이 정상적으로 기록되는지 확인합니다.

- Train/Valid total loss
- mAP50:95, AP50, AP75
- Precision, Recall, F1
- Tiny/Small person recall
- Gradient norm과 warning count
- Learning rate와 AMP scale
- 처리 속도, epoch 시간, peak VRAM

다음 상황은 바로 중단 원인으로 보지 않습니다.

- 일부 epoch의 `train_gradient_norm=inf`
- `Gradient norm exceeds 100` 경고
- AMP scale이 감소한 뒤 학습이 계속되는 상황

다만 total loss가 NaN/Inf가 되거나 CUDA OOM, traceback이 발생하면 오류입니다.

## 9. 결과 저장 위치

`last.pt`로 실행하면 데이터 프로젝트 아래에 생성됩니다.

```text
results/training/results4_fpn48_continued_from_epoch48/
├─ checkpoints/
│  ├─ best.pt
│  └─ last.pt
├─ config.json
├─ device.json
├─ history.csv
├─ warnings.log
└─ tensorboard/
```

새 결과의 `best.pt`는 이어학습 전체에서 가장 높은 validation mAP50:95 기준입니다. 기존 checkpoint가 가지고 있던 최고값 `0.252478`보다 높아야 새 `best.pt`가 저장됩니다.

## 10. AI 팀에 다시 전달할 파일

학습이 끝나거나 중간에 멈춰도 다음 폴더 전체를 압축해서 전달합니다.

```text
results/training/results4_fpn48_continued_from_epoch48/
```

최소 필수 파일:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `history.csv`
- `config.json`
- `device.json`
- `warnings.log`

함께 알려줄 내용:

- 사용한 노트북 번호
- 시작·종료 시각
- 실행 명령
- 중간에 절전·재부팅·창 종료가 있었는지
- batch size나 환경을 변경했는지
- Python, PyTorch, CUDA 버전

## 11. 결과 해석 시 주의사항

- mAP 하나만 보고 최종 모델을 결정하지 않습니다.
- 캘리브레이션용 위치 계산에는 Recall과 bbox 품질, 작은 사람 Recall도 중요합니다.
- Train loss가 계속 낮아져도 Valid mAP가 떨어지면 과적합일 수 있습니다.
- `best.pt`의 `best`는 F1이 아닙니다.
- 다른 데이터셋을 사용한 추가학습 결과는 기존 grouped Valid 결과와 직접 비교하면 안 됩니다.
- Test split을 반복해서 보면 최종 평가의 독립성이 사라집니다.

## 12. 캘리브레이션 연결 시 주의사항

학습 checkpoint는 곧바로 C++이나 Raspberry Pi에서 실행하는 파일이 아닙니다. 최종 후보가 정해지면 AI 팀에서 FP32 ONNX export와 INT8 양자화·정확도 검증을 수행합니다.

캘리브레이션 팀이 사용할 검출 결과에는 다음 정보가 필요합니다.

- `frame_id`
- 실제 프레임 촬영 `timestamp`
- 원본 640×480 기준 bbox
- confidence
- bbox 하단 중심점
- bbox가 영상 경계에 닿았는지 여부
- 사용 모델과 threshold 정보

bbox 하단이 프레임 아래 경계에 닿거나 사람이 부분적으로 잘린 경우 하단 중심점을 실제 발 위치로 확정하면 안 됩니다.

## 13. 파일 무결성

전달 후 다음 파일의 SHA-256을 확인할 수 있습니다.

```text
SHA256SUMS.txt
```

핵심 checkpoint:

```text
results4_fpn48_best.pt
SHA-256: c410e03fe58e362bea131c9b4ad70edb08886a0636f8616412c3a85028b28b36

results4_fpn48_last.pt
SHA-256: 206ff674952b149b10f0a1f30332316d6b458af59a2fa421f52cb346b0bfa23e
```
