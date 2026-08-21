from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox_bgr(frame: np.ndarray, width: int, height: int):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    px = (width - nw) // 2
    py = (height - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    return x, scale, px, py


def unletterbox_boxes(boxes: np.ndarray, scale: float, px: int, py: int,
                      original_w: int, original_h: int) -> np.ndarray:
    """Map model-input pixel coordinates back onto the original camera frame."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - px) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - py) / scale

    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, max(0, original_w - 1))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, max(0, original_h - 1))
    return out


def nms(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        iw = np.maximum(0, xx2 - xx1)
        ih = np.maximum(0, yy2 - yy1)
        inter = iw * ih
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        order = order[1:][iou <= iou_thr]

    return keep


class Model:
    def __init__(self, name: str, path: Path, threads: int):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1

        self.name = name
        self.path = path
        self.session = ort.InferenceSession(
            str(path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.height = int(shape[2])
        self.width = int(shape[3])

    def run(self, frame, conf, iou):
        original_h, original_w = frame.shape[:2]

        t0 = time.perf_counter()
        x, scale, px, py = letterbox_bgr(frame, self.width, self.height)
        t1 = time.perf_counter()

        boxes, scores = self.session.run(None, {self.input_name: x})
        t2 = time.perf_counter()

        boxes = boxes[0]
        scores = scores[0]

        mask = scores >= conf
        boxes = boxes[mask]
        scores = scores[mask]

        keep = nms(boxes, scores, iou)
        boxes = boxes[keep] if keep else np.empty((0, 4), dtype=np.float32)
        scores = scores[keep] if keep else np.empty((0,), dtype=np.float32)

        # IMPORTANT:
        # ONNX boxes are in each model's letterboxed input coordinate system.
        # Convert them back to the original camera-frame coordinate system
        # before drawing/comparing.
        boxes = unletterbox_boxes(
            boxes,
            scale=scale,
            px=px,
            py=py,
            original_w=original_w,
            original_h=original_h,
        )

        t3 = time.perf_counter()

        return {
            "boxes": boxes,
            "scores": scores,
            "pre_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }


def draw_panel(frame, result, model_name, conf):
    img = frame.copy()

    for box, score in zip(result["boxes"], result["scores"]):
        x1, y1, x2, y2 = [int(round(v)) for v in box]

        # Skip degenerate boxes after clipping.
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{score:.2f}",
            (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    title = (
        f"{model_name} | conf={conf:.2f} | det={len(result['scores'])} "
        f"| infer={result['infer_ms']:.1f}ms"
    )
    cv2.putText(
        img,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def open_source(args):
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, None

        cap.release()
        return

    src = Path(args.source)

    if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(src))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {src}")
        yield frame, src.name
        return

    if src.is_file():
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, f"frame_{idx:06d}"
            idx += 1

        cap.release()
        return

    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            frame = cv2.imread(str(p))
            if frame is not None:
                yield frame, p.name
        return

    raise FileNotFoundError(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True, help="models.json")
    ap.add_argument("--source", type=str, default="")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.17)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("compare_results"))
    args = ap.parse_args()

    if args.camera is None and not args.source:
        raise SystemExit("Use --source <video/image/folder> or --camera 0")

    cfg = json.loads(args.models.read_text(encoding="utf-8"))

    models = []
    for item in cfg["models"]:
        p = Path(item["path"])
        if not p.is_absolute():
            p = (args.models.parent / p).resolve()

        if not p.exists():
            raise FileNotFoundError(f"{item['name']}: {p}")

        models.append(Model(item["name"], p, args.threads))

    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    writer = None

    print("=" * 120)
    print("RC PERSON DETECTOR — SIDE BY SIDE MODEL COMPARISON")
    print("=" * 120)

    for m in models:
        print(f"{m.name:28s} input={m.width}x{m.height} file={m.path}")

    print(f"conf={args.conf} iou={args.iou} threads={args.threads}")
    print()

    for idx, (frame, label) in enumerate(open_source(args)):
        if idx >= args.max_frames:
            break

        panels = []

        for m in models:
            r = m.run(frame, args.conf, args.iou)

            rows.append({
                "frame": idx,
                "label": label or "",
                "model": m.name,
                "input_w": m.width,
                "input_h": m.height,
                "detections": len(r["scores"]),
                "max_score": float(r["scores"].max()) if len(r["scores"]) else 0.0,
                "pre_ms": r["pre_ms"],
                "infer_ms": r["infer_ms"],
                "post_ms": r["post_ms"],
                "total_ms": r["total_ms"],
                "fps_equiv": 1000.0 / max(r["total_ms"], 1e-9),
            })

            panels.append(draw_panel(frame, r, m.name, args.conf))

        h = min(p.shape[0] for p in panels)
        normalized = [
            cv2.resize(p, (round(p.shape[1] * h / p.shape[0]), h))
            for p in panels
        ]
        mosaic = np.hstack(normalized)

        if args.save_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(args.output / "comparison.mp4"),
                    fourcc,
                    20.0,
                    (mosaic.shape[1], mosaic.shape[0]),
                )
            writer.write(mosaic)

        if args.show:
            cv2.imshow("model comparison", mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        if idx % 20 == 0:
            print(f"frame {idx}")

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("No frames processed.")

    csv_path = args.output / "comparison_per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for m in models:
        subset = [r for r in rows if r["model"] == m.name]

        summary.append({
            "model": m.name,
            "frames": len(subset),
            "mean_detections": float(np.mean([r["detections"] for r in subset])),
            "total_detections": int(sum(r["detections"] for r in subset)),
            "mean_infer_ms": float(np.mean([r["infer_ms"] for r in subset])),
            "p95_infer_ms": float(np.percentile([r["infer_ms"] for r in subset], 95)),
            "mean_total_ms": float(np.mean([r["total_ms"] for r in subset])),
            "mean_fps_equiv": float(np.mean([r["fps_equiv"] for r in subset])),
        })

    sum_path = args.output / "comparison_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print("\n" + "-" * 120)
    for s in summary:
        print(
            f"{s['model']:28s} frames={s['frames']:4d} "
            f"infer={s['mean_infer_ms']:7.2f}ms "
            f"p95={s['p95_infer_ms']:7.2f} "
            f"total={s['mean_total_ms']:7.2f}ms "
            f"fps={s['mean_fps_equiv']:6.2f} "
            f"dets={s['total_detections']}"
        )

    print("-" * 120)
    print("saved:", csv_path)
    print("saved:", sum_path)

    if args.save_video:
        print("saved:", args.output / "comparison.mp4")


if __name__ == "__main__":
    main()
