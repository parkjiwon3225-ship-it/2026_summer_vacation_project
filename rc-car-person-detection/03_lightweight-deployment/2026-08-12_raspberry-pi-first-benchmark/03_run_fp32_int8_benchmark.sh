#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

.venv/bin/python pi_first_benchmark.py \
  --camera 0 \
  --camera-backend picamera2 \
  --capture-width 640 \
  --capture-height 480 \
  --capture-fps 30 \
  --duration 180 \
  --compare-frames 300 \
  --threads 4
