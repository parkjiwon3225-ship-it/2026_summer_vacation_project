# 학교 GPU 노트북 학습 실행 안내

## 1. 프로젝트 복사

`rc_person_detector` 폴더 전체를 학교 노트북의 짧은 영문 경로에 복사한다.

예시:

```text
C:\projects\rc_person_detector
```

`archive`와 `data/processed/v1_grouped`가 포함되어 있는지 확인한다.

## 2. Conda 환경 생성

Anaconda Prompt에서 프로젝트 폴더로 이동한 뒤 실행한다.

```powershell
conda env create -f environment-school.yml
conda activate rc-person-detector
python -m pip install -e .
python -m ipykernel install --user --name rc-person-detector --display-name "RC Person Detector"
```

환경 파일은 Python 3.11과 CUDA 12.1용 PyTorch를 기준으로 한다. 학교 노트북의 NVIDIA
driver가 CUDA 12.1 runtime을 지원하지 않거나 Conda가 패키지를 찾지 못하면 임의로 다른
조합을 설치하지 말고 GPU/driver 정보를 먼저 기록한다.

## 3. GPU 사전 확인

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

`torch.cuda.is_available()`이 `True`가 아니면 전체 학습을 시작하지 않는다.

## 4. 파이프라인 검사

```powershell
python scripts\13_test_training_step.py
python scripts\14_test_inference.py
python scripts\15_mini_overfit.py --device cuda
```

세 검사가 모두 PASS여야 전체 학습으로 넘어간다.

## 5. Jupyter 실행

```powershell
jupyter lab
```

`notebooks/01_train_v1.ipynb`를 열고 커널을 `RC Person Detector`로 선택한다.

## 6. 전체 학습

기본 설정은 `configs/train_v1.json`에 있다. 기본 batch size는 8이다. VRAM 부족 시 먼저
batch size를 4로 낮춘다. 그래도 부족할 때 2로 낮추고 gradient accumulation을 늘린다.

터미널 직접 실행:

```powershell
python scripts\16_train.py
```

batch size를 일시적으로 바꾸는 예:

```powershell
python scripts\16_train.py --batch-size 4
```

## 7. 중단 후 재개

매 epoch 종료 후 다음 체크포인트가 원자적으로 갱신된다.

```text
results/training/person_detector_v1/checkpoints/last.pt
results/training/person_detector_v1/checkpoints/best.pt
```

재개 명령:

```powershell
python scripts\16_train.py --resume results\training\person_detector_v1\checkpoints\last.pt
```

`best.pt`는 validation loss가 가장 낮은 모델이고, `last.pt`는 가장 최근 상태다.

## 8. 결과 파일

```text
results/training/person_detector_v1/config.json
results/training/person_detector_v1/device.json
results/training/person_detector_v1/history.csv
results/training/person_detector_v1/checkpoints/best.pt
results/training/person_detector_v1/checkpoints/last.pt
```

`device.json`에는 실제 GPU 이름, VRAM, CUDA build와 PyTorch 버전이 저장된다.
