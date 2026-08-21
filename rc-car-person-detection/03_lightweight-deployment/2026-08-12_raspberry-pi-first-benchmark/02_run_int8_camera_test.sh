#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

.venv/bin/python pi_int8_camera_test.py \
  --duration 600 \
  --threshold 0.20 \
  --nms 0.20 \
  --threads 2 \
  --max-temperature 75
