from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


INPUT_W = 640
INPUT_H = 480
FPN_CHANNELS = 48
BACKBONE_EXPANSION = 2.0
STRIDES = (("p2", 4), ("p3", 8), ("p4", 16), ("p5", 32))
EXPECTED_POINTS = 160 * 120 + 80 * 60 + 40 * 30 + 20 * 15  # 25500


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_project_root() -> Path:
    start = package_root()
    candidates = [start, *start.parents]
    for p in candidates:
        if (p / "src" / "rc_detector" / "model.py").is_file():
            return p
    raise FileNotFoundError(
        "rc_person_detector project root를 찾지 못했습니다. "
        "results27_lightweight_pipeline 폴더를 rc_person_detector 프로젝트 루트에 복사하세요."
    )


def add_project_src() -> Path:
    root = find_project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def torch_load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_source_model():
    add_project_src()
    from rc_detector.model import PersonDetector

    ckpt_path = package_root() / "source" / "results27_best_e52.pt"
    ckpt = torch_load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {})

    epoch = int(ckpt.get("epoch", -1))
    best_map = float(ckpt.get("best_map50_95", float("nan")))
    width = int(cfg.get("image_width", -1))
    height = int(cfg.get("image_height", -1))
    fpn = int(cfg.get("fpn_channels", -1))
    expansion = float(cfg.get("backbone_expansion", -1))

    expected = {
        "epoch": 52,
        "image_width": INPUT_W,
        "image_height": INPUT_H,
        "fpn_channels": FPN_CHANNELS,
        "backbone_expansion": BACKBONE_EXPANSION,
    }
    actual = {
        "epoch": epoch,
        "image_width": width,
        "image_height": height,
        "fpn_channels": fpn,
        "backbone_expansion": expansion,
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise RuntimeError(f"results.27 source mismatch: {key}={actual[key]} expected={value}")

    model = PersonDetector(
        fpn_channels=FPN_CHANNELS,
        backbone_expansion=BACKBONE_EXPANSION,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt, ckpt_path


class ExportDetector(torch.nn.Module):
    """Convert internal pyramid dictionaries into deployment tensors.

    Outputs:
      boxes:  [B, 25500, 4] absolute xyxy coordinates in 640x480 model space
      scores: [B, 25500]    sigmoid(class) * sigmoid(quality)
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        predictions = self.model(images)
        all_boxes = []
        all_scores = []

        for level, stride in STRIDES:
            out = predictions[level]
            class_logits = out["class_logits"]
            quality_logits = out["quality_logits"]
            distances = out["distances"]

            batch = class_logits.shape[0]
            height = class_logits.shape[-2]
            width = class_logits.shape[-1]

            xs = (torch.arange(width, device=images.device, dtype=distances.dtype) + 0.5) * stride
            ys = (torch.arange(height, device=images.device, dtype=distances.dtype) + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            px = grid_x.reshape(1, -1)
            py = grid_y.reshape(1, -1)

            d = distances.permute(0, 2, 3, 1).reshape(batch, -1, 4)
            boxes = torch.stack(
                (
                    px - d[:, :, 0],
                    py - d[:, :, 1],
                    px + d[:, :, 2],
                    py + d[:, :, 3],
                ),
                dim=2,
            )

            scores = (
                class_logits[:, 0].sigmoid().reshape(batch, -1)
                * quality_logits[:, 0].sigmoid().reshape(batch, -1)
            )
            all_boxes.append(boxes)
            all_scores.append(scores)

        return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1)


def letterbox_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(INPUT_W / width, INPUT_H / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    pad_x = (INPUT_W - resized_width) // 2
    pad_y = (INPUT_H - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (INPUT_W, INPUT_H), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def list_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


def select_evenly(items: list[Path], count: int) -> list[Path]:
    if not items:
        return []
    if count >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(i)] for i in idx]


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-7)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def detections_from_raw(
    boxes: np.ndarray,
    scores: np.ndarray,
    conf: float,
    nms_iou: float = 0.5,
    max_det: int = 100,
):
    mask = scores >= conf
    b = boxes[mask].copy()
    s = scores[mask].copy()
    if not len(s):
        return []
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, INPUT_W)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, INPUT_H)
    valid = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    b, s = b[valid], s[valid]
    if not len(s):
        return []
    if len(s) > 1000:
        top = np.argpartition(s, -1000)[-1000:]
        b, s = b[top], s[top]
    keep = nms_numpy(b, s, nms_iou)[:max_det]
    return [(b[i], float(s[i])) for i in keep]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(reference, candidate, iou_threshold: float = 0.5):
    pairs = []
    for i, (rb, _) in enumerate(reference):
        for j, (cb, _) in enumerate(candidate):
            iou = box_iou(rb, cb)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_r, used_c, matches = set(), set(), []
    for iou, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        matches.append((i, j, iou))
    missed = len(reference) - len(used_r)
    extra = len(candidate) - len(used_c)
    return matches, missed, extra
