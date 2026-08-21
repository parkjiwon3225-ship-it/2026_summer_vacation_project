#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-120}"

for MODEL in \
  models/results4_fpn48_fp32.onnx \
  models/results4_fpn48_int8_qdq_percentile.onnx \
  models/results4_fpn48_int8_qdq_minmax.onnx
do
  echo "============================================================"
  echo "Testing $MODEL for $DURATION seconds"
  echo "Use the same scene and person distance for every model."
  echo "============================================================"
  .venv/bin/python scripts/pi_model_camera_test.py \
    --model "$MODEL" \
    --duration "$DURATION" \
    --threshold 0.25 \
    --nms 0.50 \
    --threads 2 \
    --max-temperature 75
  echo "Let the Raspberry Pi cool before the next model if needed."
done
