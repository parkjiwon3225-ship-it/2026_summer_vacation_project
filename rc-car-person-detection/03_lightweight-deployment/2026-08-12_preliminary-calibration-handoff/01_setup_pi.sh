#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

sudo apt update
sudo apt install -y python3-venv python3-opencv python3-picamera2

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-pi.txt

.venv/bin/python -c "import cv2, numpy, onnxruntime, psutil, picamera2; print('SETUP PASS'); print('OpenCV', cv2.__version__); print('ONNX Runtime', onnxruntime.__version__); print('Picamera2', getattr(picamera2, '__version__', 'installed'))"
