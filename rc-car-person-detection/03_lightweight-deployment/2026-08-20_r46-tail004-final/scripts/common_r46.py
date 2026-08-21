from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

INPUT_W = 448
INPUT_H = 336
STRIDES = (("p2", 4), ("p3", 8), ("p4", 16), ("p5", 32))
EXPECTED_POINTS = 112 * 84 + 56 * 42 + 28 * 21 + 14 * 11  # 12502
EXPECTED_RUN = "r46_final448_seed15_100e"
EXPECTED_EPOCH = 25

def package_root() -> Path:
    return Path(__file__).resolve().parents[1]

def add_src() -> None:
    src = package_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_source_model():
    add_src()
    from rc_detector.model import PersonDetector
    p = package_root() / "source" / "R46_original_best.pt"
    if not p.is_file():
        raise FileNotFoundError(p)
    ckpt = torch_load(p)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError("Unsupported checkpoint payload")
    cfg = ckpt.get("config", {})
    actual = {
        "experiment_name": str(cfg.get("experiment_name", "")),
        "epoch": int(ckpt.get("epoch", -1)),
        "image_width": int(cfg.get("image_width", -1)),
        "image_height": int(cfg.get("image_height", -1)),
        "fpn_channels": int(cfg.get("fpn_channels", -1)),
        "backbone_expansion": float(cfg.get("backbone_expansion", -1)),
    }
    expected = {
        "experiment_name": EXPECTED_RUN,
        "epoch": EXPECTED_EPOCH,
        "image_width": INPUT_W,
        "image_height": INPUT_H,
    }
    for k, v in expected.items():
        if actual[k] != v:
            raise RuntimeError(f"R46 source mismatch: {k}={actual[k]!r} expected={v!r}")
    if actual["fpn_channels"] <= 0 or actual["backbone_expansion"] <= 0:
        raise RuntimeError(f"Invalid architecture metadata: {actual}")
    model = PersonDetector(
        fpn_channels=actual["fpn_channels"],
        backbone_expansion=actual["backbone_expansion"],
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt, p, actual

class ExportDetector(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
    def forward(self, images: torch.Tensor):
        predictions = self.model(images)
        all_boxes, all_scores = [], []
        for level, stride in STRIDES:
            out = predictions[level]
            class_logits = out["class_logits"]
            quality_logits = out["quality_logits"]
            distances = out["distances"]
            batch = class_logits.shape[0]
            h, w = class_logits.shape[-2:]
            xs = (torch.arange(w, device=images.device, dtype=distances.dtype) + 0.5) * stride
            ys = (torch.arange(h, device=images.device, dtype=distances.dtype) + 0.5) * stride
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            px, py = gx.reshape(1, -1), gy.reshape(1, -1)
            d = distances.permute(0, 2, 3, 1).reshape(batch, -1, 4)
            boxes = torch.stack((px-d[:,:,0], py-d[:,:,1], px+d[:,:,2], py+d[:,:,3]), dim=2)
            scores = class_logits[:,0].sigmoid().reshape(batch,-1) * quality_logits[:,0].sigmoid().reshape(batch,-1)
            all_boxes.append(boxes); all_scores.append(scores)
        return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1)

def letterbox_image(path: Path) -> np.ndarray:
    with Image.open(path) as src:
        image = src.convert("RGB")
    w, h = image.size
    scale = min(INPUT_W / w, INPUT_H / h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    px, py = (INPUT_W-rw)//2, (INPUT_H-rh)//2
    resized = image.resize((rw,rh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (INPUT_W, INPUT_H), (114,114,114))
    canvas.paste(resized, (px,py))
    arr = np.asarray(canvas, dtype=np.float32)/255.0
    return np.ascontiguousarray(np.transpose(arr,(2,0,1))[None], dtype=np.float32)

def list_images(directory: Path) -> list[Path]:
    exts={".jpg",".jpeg",".png",".bmp",".webp"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, threshold: float=0.5) -> np.ndarray:
    if boxes.size==0: return np.empty((0,),dtype=np.int64)
    x1,y1,x2,y2=boxes.T
    areas=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    order=np.argsort(scores)[::-1]; keep=[]
    while order.size:
        i=int(order[0]); keep.append(i)
        if order.size==1: break
        r=order[1:]
        xx1=np.maximum(x1[i],x1[r]); yy1=np.maximum(y1[i],y1[r]); xx2=np.minimum(x2[i],x2[r]); yy2=np.minimum(y2[i],y2[r])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/np.maximum(areas[i]+areas[r]-inter,1e-7)
        order=r[iou<=threshold]
    return np.asarray(keep,dtype=np.int64)

def detections_from_raw(boxes: np.ndarray, scores: np.ndarray, conf: float, nms_iou: float=0.5, max_det: int=100):
    mask=scores>=conf; b=boxes[mask].copy(); s=scores[mask].copy()
    if len(s)==0:return []
    b[:,[0,2]]=np.clip(b[:,[0,2]],0,INPUT_W); b[:,[1,3]]=np.clip(b[:,[1,3]],0,INPUT_H)
    valid=(b[:,2]>b[:,0])&(b[:,3]>b[:,1]); b,s=b[valid],s[valid]
    if len(s)==0:return []
    if len(s)>1000:
        top=np.argpartition(s,-1000)[-1000:]; b,s=b[top],s[top]
    keep=nms_numpy(b,s,nms_iou)[:max_det]
    return [(b[i],float(s[i])) for i in keep]

def box_iou(a: np.ndarray,b: np.ndarray)->float:
    ix1=max(float(a[0]),float(b[0])); iy1=max(float(a[1]),float(b[1])); ix2=min(float(a[2]),float(b[2])); iy2=min(float(a[3]),float(b[3]))
    inter=max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1])); ab=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    u=aa+ab-inter
    return inter/u if u>0 else 0.0

def greedy_match(reference,candidate,iou_threshold:float=0.5):
    pairs=[]
    for i,(rb,_) in enumerate(reference):
        for j,(cb,_) in enumerate(candidate):
            iou=box_iou(rb,cb)
            if iou>=iou_threshold:pairs.append((iou,i,j))
    pairs.sort(reverse=True); ur=set();uc=set();matches=[]
    for iou,i,j in pairs:
        if i in ur or j in uc:continue
        ur.add(i);uc.add(j);matches.append((i,j,iou))
    return matches,len(reference)-len(ur),len(candidate)-len(uc)
