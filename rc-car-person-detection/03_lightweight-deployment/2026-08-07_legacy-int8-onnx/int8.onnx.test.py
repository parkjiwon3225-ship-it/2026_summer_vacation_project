import os
import onnxruntime as ort
import numpy as np
import cv2


MODEL = "person_detector_v2_best_int8.onnx"

IMAGE_DIR = "./archive/test/test"


session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)


input_name = session.get_inputs()[0].name

print("Input:", input_name)
print("Shape:", session.get_inputs()[0].shape)


# 이미지 자동 선택
image_files = []

for f in os.listdir(IMAGE_DIR):
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        image_files.append(f)


if len(image_files) == 0:
    raise Exception("No images found")


image_path = os.path.join(
    IMAGE_DIR,
    image_files[0]
)


print("Test Image:", image_path)


img = cv2.imread(
    image_path
)


if img is None:
    raise Exception(
        "Image load failed"
    )


img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


img = cv2.resize(
    img,
    (320,240)
)


img = img.astype(
    np.float32
) / 255.0


img = np.transpose(
    img,
    (2,0,1)
)


img = np.expand_dims(
    img,
    axis=0
)


output = session.run(
    None,
    {
        input_name: img
    }
)


print("Output shape:")
print(output[0].shape)

print("Inference OK")