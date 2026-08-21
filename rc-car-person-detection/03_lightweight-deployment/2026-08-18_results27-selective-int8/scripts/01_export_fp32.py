from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    EXPECTED_POINTS,
    INPUT_H,
    INPUT_W,
    ExportDetector,
    list_images,
    find_project_root,
    letterbox_image,
    load_source_model,
    package_root,
    select_evenly,
)


def require_onnx_tools():
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        print("\n[ERROR] ONNX 도구가 없습니다:", e)
        print("아래 명령을 현재 rc-person-detector 환경에서 실행하세요:")
        print("  python -m pip install onnx==1.22.0 onnxruntime==1.28.0")
        sys.exit(2)


def main():
    require_onnx_tools()
    import onnx
    import onnxruntime as ort

    root = find_project_root()
    out_dir = package_root() / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results27_640_fp32.onnx"

    model, ckpt, ckpt_path = load_source_model()
    wrapper = ExportDetector(model).eval()

    dummy = torch.zeros((1, 3, INPUT_H, INPUT_W), dtype=torch.float32)
    with torch.no_grad():
        boxes, scores = wrapper(dummy)
    if tuple(boxes.shape) != (1, EXPECTED_POINTS, 4):
        raise RuntimeError(f"Unexpected boxes shape: {tuple(boxes.shape)}")
    if tuple(scores.shape) != (1, EXPECTED_POINTS):
        raise RuntimeError(f"Unexpected scores shape: {tuple(scores.shape)}")

    print("=" * 78)
    print("RESULTS.27 FP32 ONNX EXPORT")
    print("=" * 78)
    print("checkpoint :", ckpt_path)
    print("epoch      :", ckpt["epoch"])
    print("mAP50-95   :", ckpt["best_map50_95"])
    print("input      :", (1, 3, INPUT_H, INPUT_W))
    print("boxes      :", tuple(boxes.shape))
    print("scores     :", tuple(scores.shape))
    print("output     :", out_path)

    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["boxes", "scores"],
        dynamic_axes=None,
    )
    print(f"export_sec : {time.perf_counter() - t0:.2f}")

    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(out_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    print("ORT input  :", session.get_inputs()[0].name, session.get_inputs()[0].shape)
    print("ORT outputs:", [(o.name, o.shape) for o in session.get_outputs()])

    # Real-image equivalence check when dataset is available.
    valid_dir = root / "data" / "processed" / "v1_grouped" / "valid" / "images"
    if valid_dir.is_dir():
        samples = select_evenly(list_images(valid_dir), 3)
        max_score_diff = 0.0
        mean_score_diffs = []
        max_box_diff = 0.0
        mean_box_diffs = []
        for p in samples:
            x = letterbox_image(p)
            with torch.no_grad():
                pt_b, pt_s = wrapper(torch.from_numpy(x))
            ort_b, ort_s = session.run(None, {"images": x})
            bd = np.abs(pt_b.numpy() - ort_b)
            sd = np.abs(pt_s.numpy() - ort_s)
            max_box_diff = max(max_box_diff, float(bd.max()))
            max_score_diff = max(max_score_diff, float(sd.max()))
            mean_box_diffs.append(float(bd.mean()))
            mean_score_diffs.append(float(sd.mean()))
        print("\nPyTorch -> FP32 ONNX real-image raw equivalence")
        print("score max abs diff :", max_score_diff)
        print("score mean abs diff:", float(np.mean(mean_score_diffs)))
        print("box max abs diff   :", max_box_diff)
        print("box mean abs diff  :", float(np.mean(mean_box_diffs)))
    else:
        print("\n[WARN] valid images not found; real-image equivalence check skipped.")

    print("\n[OK] FP32 ONNX export complete.")
    print(f"size_mb: {out_path.stat().st_size / 1024 / 1024:.3f}")


if __name__ == "__main__":
    main()
