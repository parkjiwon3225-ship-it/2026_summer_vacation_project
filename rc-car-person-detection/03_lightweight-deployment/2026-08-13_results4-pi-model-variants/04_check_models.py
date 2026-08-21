from pathlib import Path

import numpy as np
import onnxruntime as ort


models = sorted((Path(__file__).resolve().parent / "models").glob("*.onnx"))
if not models:
    raise RuntimeError("No ONNX models found")

dummy = np.zeros((1, 3, 240, 320), dtype=np.float32)
for model in models:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: dummy})
    shapes = [tuple(value.shape) for value in outputs]
    if shapes != [(1, 6380, 4), (1, 6380)]:
        raise RuntimeError(f"Unexpected outputs for {model.name}: {shapes}")
    print(f"PASS {model.name}: {model.stat().st_size / 1024 / 1024:.3f} MiB {shapes}")
