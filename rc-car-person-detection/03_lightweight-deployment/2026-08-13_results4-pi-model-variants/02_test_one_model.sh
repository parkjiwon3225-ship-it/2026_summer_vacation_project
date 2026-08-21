#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: ./02_test_one_model.sh models/MODEL.onnx [duration_seconds]"
  exit 2
fi

MODEL="$1"
DURATION="${2:-180}"

.venv/bin/python scripts/pi_model_camera_test.py \
  --model "$MODEL" \
  --duration "$DURATION" \
  --threshold 0.25 \
  --nms 0.50 \
  --threads 2 \
  --max-temperature 75
