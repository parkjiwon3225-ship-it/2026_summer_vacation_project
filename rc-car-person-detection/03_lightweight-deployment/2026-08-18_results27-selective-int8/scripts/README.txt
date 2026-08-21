Copy both .py files into results27_lightweight_pipeline\scripts\
Run:
python scripts\04_quantize_int8_round2.py --calibration-samples 400
python scripts\05_validate_int8_round2.py --samples 60 --threads 2
