from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def postprocess(boxes_np, scores_np, nms, threshold=0.001, max_detections=100):
    from rc_detector.inference import pure_torch_nms

    boxes = torch.from_numpy(boxes_np[0]).float()
    scores = torch.from_numpy(scores_np[0]).float()
    keep = scores >= threshold
    boxes, scores = boxes[keep], scores[keep]
    if len(scores) > 1000:
        scores, indices = scores.topk(1000)
        boxes = boxes[indices]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, 320)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, 240)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    selected = pure_torch_nms(boxes, scores, nms)[:max_detections]
    return {"boxes": boxes[selected], "scores": scores[selected]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from rc_detector.dataset import PersonDetectionDataset
    from rc_detector.metrics import DetectionEvaluator

    dataset = PersonDetectionDataset(args.dataset, "valid", (320, 240), augment=False)
    report = []
    for model_path in sorted(args.models.glob("*.onnx")):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        evaluator = DetectionEvaluator(operating_score_threshold=0.25)
        times = []
        for index in range(len(dataset)):
            image, target = dataset[index]
            array = np.ascontiguousarray(image.numpy()[None], dtype=np.float32)
            started = time.perf_counter()
            boxes, scores = session.run(None, {input_name: array})
            times.append((time.perf_counter() - started) * 1000.0)
            prediction = postprocess(boxes, scores, 0.5)
            evaluator.update([prediction], [target])
            if (index + 1) % 250 == 0:
                print(model_path.name, index + 1, "/", len(dataset), flush=True)
        metrics = evaluator.compute()
        report.append({
            "model": model_path.name,
            "size_mib": model_path.stat().st_size / (1024 ** 2),
            "mean_inference_ms_local_2threads": float(np.mean(times)),
            "p95_inference_ms_local_2threads": float(np.percentile(times, 95)),
            **{key: value for key, value in metrics.items() if key != "size_recall"},
            "tiny_recall": metrics["size_recall"]["tiny_lt16"]["recall"],
            "small_recall": metrics["size_recall"]["small_16_32"]["recall"],
            "medium_recall": metrics["size_recall"]["medium_32_96"]["recall"],
            "large_recall": metrics["size_recall"]["large_ge96"]["recall"],
        })
        print(json.dumps(report[-1], indent=2), flush=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
