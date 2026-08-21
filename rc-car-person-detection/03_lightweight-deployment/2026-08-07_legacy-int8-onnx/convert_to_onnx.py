import torch
import torch.nn as nn

from person_detector_model import PersonDetectorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

NUM_ANCHORS = 5

BACKBONE_CHANNELS = (
    32,
    64,
    128,
    128,
    256,
    256,
    256
)

MODEL_PATH = "person_detector_v2_best.pth"

model = PersonDetectorV2(
    backbone_channels=BACKBONE_CHANNELS,
    num_anchors=NUM_ANCHORS
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

dummy_input = torch.randn(
    1,
    3,
    IMAGE_HEIGHT,
    IMAGE_WIDTH
).to(DEVICE)

with torch.no_grad():
    output = model(dummy_input)

print("Input :", dummy_input.shape)
print("Output:", output.shape)

ONNX_PATH = "person_detector_v2_best.onnx"

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,

    input_names=[
        "input"
    ],

    output_names=[
        "output"
    ],

    dynamic_axes={
        "input": {
            0: "batch"
        },
        "output": {
            0: "batch"
        }
    }
)