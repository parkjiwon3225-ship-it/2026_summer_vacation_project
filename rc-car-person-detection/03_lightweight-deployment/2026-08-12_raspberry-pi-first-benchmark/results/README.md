# Raspberry Pi INT8 camera test results

Raspberry Pi, OV5647 camera and the temporary INT8 ONNX person detector were tested on 2026-08-12. No camera frames or videos are stored here; only numeric results are retained.

## Directory contents

Each timestamped run contains:

- `summary.json`: run configuration and aggregate metrics
- `frame_metrics.csv`: frame-by-frame latency, temperature and detection counts
- `detections.csv`: confidence and box-size data for individual detections

## Run index

| Run | Threads | Status | Stop reason | Frames | Detections | Mean inference | P95 inference | Mean sensor-to-result | Peak temp. |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `20260812_150616` | 4 | THERMAL_STOP | 78.4 C reached | 666 | 544 | 62.06 ms | 83.18 ms | 111.92 ms | 78.39 C |
| `20260812_150847` | 4 | THERMAL_STOP | 78.4 C reached | 387 | 674 | 60.99 ms | 77.65 ms | 110.36 ms | 78.39 C |
| `20260812_151407` | 2 | PASS | user pressed Q | 1,099 | 2,983 | 90.22 ms | 94.39 ms | 136.20 ms | 71.58 C |
| `20260812_151849` | 3 | THERMAL_STOP | 75.5 C reached | 611 | 1,232 | 66.96 ms | 75.17 ms | 115.13 ms | 75.47 C |
| `20260812_152543` | 2 | PASS | user pressed Q | 699 | 2,001 | 89.95 ms | 93.99 ms | 137.38 ms | 68.65 C |
| `20260812_153248` | 2 | PASS | duration completed | 1,098 | 1,402 | 90.63 ms | 95.65 ms | 137.75 ms | 72.55 C |

All runs used confidence threshold `0.20` and NMS threshold `0.20`.

## Interpretation

- Four threads were fastest but reached 78.4 C and stopped thermally.
- Three threads also reached the 75 C safety threshold.
- Two threads completed or remained below the thermal cutoff in the recorded runs, so it is the current no-cooling baseline.
- The final `20260812_153248` run used the corrected `RGB888` camera format. Color and bounding-box display were visually confirmed as normal.
- Detection counts across runs are not accuracy scores. Scenes, person movement, test duration and manual labels differed, and no ground-truth annotations were collected.
- Final model accuracy must be measured on the grouped validation/test dataset; these Raspberry Pi runs measure deployment behavior, latency and temperature.

## Current deployment baseline

- Model: temporary INT8 ONNX model
- ONNX Runtime threads: `2`
- Confidence threshold: `0.20` for exploratory visual testing
- NMS threshold: `0.20`
- Camera format: `RGB888`
- Thermal guard: `75 C`
- Cooling: no heatsink or fan during these tests

The thresholds are provisional and must be tuned again after the final trained model is exported.
