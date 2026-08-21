RESULTS.27 INT8 ROUND 4 — BACKBONE LAYER SENSITIVITY

Purpose
Round 3 showed that even backbone-only PTQ changed detections substantially.
Round 4 quantizes contiguous quarters of backbone Conv nodes to locate the sensitive region.

Candidates
- q1_early_only
- q2_only
- q3_only
- q4_late_only
- q1_q2_prefix50
- q3_q4_suffix50

Install
Copy 08_quantize_int8_round4.py and 09_validate_int8_round4.py into the existing scripts folder.

Run
python scripts\08_quantize_int8_round4.py --calibration-samples 96
python scripts\09_validate_int8_round4.py --samples 100 --threads 2
