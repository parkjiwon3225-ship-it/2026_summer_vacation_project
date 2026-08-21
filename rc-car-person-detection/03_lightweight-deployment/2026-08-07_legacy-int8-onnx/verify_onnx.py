import onnx
import onnxruntime as ort
import numpy as np


MODEL_PATH = "person_detector_v2_best.onnx"


# -----------------------------
# ONNX 파일 검사
# -----------------------------

model = onnx.load(MODEL_PATH)

onnx.checker.check_model(model)

print("ONNX model check : OK")


# -----------------------------
# ONNX Runtime 실행
# -----------------------------

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


print("\nInput name :", input_name)
print("Output name:", output_name)


print("\nInput shape:")
print(session.get_inputs()[0].shape)

print("\nOutput shape:")
print(session.get_outputs()[0].shape)


# -----------------------------
# Dummy inference
# -----------------------------

dummy_input = np.random.randn(
    1,
    3,
    240,
    320
).astype(np.float32)


outputs = session.run(
    [output_name],
    {
        input_name: dummy_input
    }
)


result = outputs[0]


print("\nInference result:")
print(result.shape)

print("\nONNX inference : SUCCESS")