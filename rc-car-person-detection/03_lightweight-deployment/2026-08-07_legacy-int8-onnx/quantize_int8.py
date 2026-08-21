import os
import cv2
import numpy as np
import onnxruntime as ort

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType
)


ONNX_MODEL = "person_detector_v2_best.onnx"
INT8_MODEL = "person_detector_v2_best_int8.onnx"

IMAGE_DIR = "./archive/train"

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


# ONNX 입력 이름 자동 확인
session = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name

print("ONNX Input Name:", INPUT_NAME)
print(
    "ONNX Input Shape:",
    session.get_inputs()[0].shape
)


class CalibrationReader(CalibrationDataReader):

    def __init__(self, image_dir):

        self.data = []

        image_files = []


        # 하위 폴더까지 이미지 검색
        for root, dirs, files in os.walk(image_dir):

            for file in files:

                if file.lower().endswith(
                    (".jpg", ".png", ".jpeg")
                ):
                    image_files.append(
                        os.path.join(root, file)
                    )


        print(
            "Found images:",
            len(image_files)
        )


        count = 0


        for path in image_files:

            img = cv2.imread(path)

            if img is None:
                continue


            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )


            img = cv2.resize(
                img,
                (IMAGE_WIDTH, IMAGE_HEIGHT)
            )


            img = img.astype(
                np.float32
            ) / 255.0


            img = np.transpose(
                img,
                (2, 0, 1)
            )


            img = np.expand_dims(
                img,
                axis=0
            )


            self.data.append(
                {
                    INPUT_NAME: img
                }
            )


            count += 1


            # calibration 100장 사용
            if count >= 100:
                break


        print(
            "Calibration images:",
            len(self.data)
        )


        self.iterator = iter(self.data)



    def get_next(self):

        return next(
            self.iterator,
            None
        )



print("===================")
print("INT8 Start")
print("===================")


reader = CalibrationReader(
    IMAGE_DIR
)


quantize_static(
    model_input=ONNX_MODEL,
    model_output=INT8_MODEL,
    calibration_data_reader=reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)


print("===================")
print("INT8 Complete")
print(INT8_MODEL)