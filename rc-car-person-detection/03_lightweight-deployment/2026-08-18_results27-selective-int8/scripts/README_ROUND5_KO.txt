RESULTS.27 INT8 ROUND 5 — SAFE BACKBONE MIX

Round 4 finding
- Q1 (early backbone) is highly PTQ-sensitive: reject.
- Q2, Q3, Q4 individually preserve ~97-98%.
- Q3+Q4 preserves ~96-97% and reduces model size to ~0.729 MB.

Round 5 goal
Keep Q1 fully FP32 and test combinations of the safe regions:
- q2_q3
- q2_q4
- q2_q3_q4

Install
Copy both scripts into the existing results27_lightweight_pipeline\scripts folder.

Run
python scripts\10_quantize_int8_round5.py --calibration-samples 128
python scripts\11_validate_int8_round5.py --samples 120 --threads 2

The validation script also includes the Round4 q3+q4 candidate automatically when it exists.
