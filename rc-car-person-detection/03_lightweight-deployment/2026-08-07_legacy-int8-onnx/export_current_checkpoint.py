"""
학습 중인 노트북은 그대로 두고, 새 Jupyter 노트북(또는 새 커널)에서
이 코드를 실행하면 현재까지의 best 체크포인트를 카메라 연동용 .pt로
바로 내보낼 수 있습니다. (학습에 전혀 영향 주지 않습니다)

사용법: 아래 CHECKPOINT_PATH, BACKBONE_CHANNELS, NUM_ANCHORS 값만
실제 학습 노트북의 "⚙️ 실험 설정" 셀 값과 동일하게 맞춘 뒤 실행하세요.
"""
import torch
import torch.nn as nn
import json
import os

# ── 아래 값들을 학습 노트북의 config와 동일하게 맞추세요 ──
CHECKPOINT_PATH = "checkpoints/balanced/person_detector_v2_best.pth"
OUTPUT_DIR = "results/balanced"
IMAGE_WIDTH, IMAGE_HEIGHT = 320, 240
# ────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, groups=1):
        super().__init__()
        padding = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU6(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.depthwise = ConvBNAct(cin, cin, k=3, stride=stride, groups=cin)
        self.pointwise = ConvBNAct(cin, cout, k=1, stride=1)
    def forward(self, x): return self.pointwise(self.depthwise(x))


class Backbone(nn.Module):
    def __init__(self, channels):
        super().__init__()
        stem_c, dw1_c, dw2_c, *refine_c = channels
        self.stem = ConvBNAct(3, stem_c, k=3, stride=2)
        self.down1 = DepthwiseSeparable(stem_c, dw1_c, stride=2)
        self.down2 = DepthwiseSeparable(dw1_c, dw2_c, stride=2)
        layers, prev = [], dw2_c
        for c in refine_c:
            layers.append(DepthwiseSeparable(prev, c, stride=1)); prev = c
        self.refine = nn.Sequential(*layers)
        self.out_channels = prev
    def forward(self, x): return self.refine(self.down2(self.down1(self.stem(x))))


class MultiAnchorHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.num_anchors = num_anchors
        self.refine = DepthwiseSeparable(in_channels, in_channels, stride=1)
        self.predict = nn.Conv2d(in_channels, num_anchors * 5, kernel_size=1)
    def forward(self, x):
        x = self.predict(self.refine(x))
        b, _, h, w = x.shape
        return x.view(b, self.num_anchors, 5, h, w)


class PersonDetectorV2(nn.Module):
    def __init__(self, backbone_channels, num_anchors):
        super().__init__()
        self.backbone = Backbone(backbone_channels)
        self.head = MultiAnchorHead(self.backbone.out_channels, num_anchors)
    def forward(self, x): return self.head(self.backbone(x))


# ── 체크포인트 로드 (학습 중인 파일을 읽기만 하므로 안전합니다) ──
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
print(f"체크포인트 epoch={ckpt['epoch']}, best_f1={ckpt['best_f1']:.4f}")

model = PersonDetectorV2(
    backbone_channels=tuple(ckpt["backbone_channels"]),
    num_anchors=ckpt["num_anchors"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

example_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
script_model = torch.jit.trace(model, example_input)

script_path = os.path.join(OUTPUT_DIR, f"person_detector_v2_epoch{ckpt['epoch']}_script.pt")
script_model.save(script_path)

anchors_path = os.path.join(OUTPUT_DIR, f"anchors_epoch{ckpt['epoch']}.json")
with open(anchors_path, "w") as f:
    json.dump({
        "anchors_wh": ckpt["anchors_wh"],
        "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
        "grid_w": ckpt["grid_w"], "grid_h": ckpt["grid_h"],
        "conf_threshold": ckpt.get("best_threshold", 0.4),
    }, f, indent=2)

print(f"✅ .pt 내보내기 완료: {script_path}")
print(f"✅ anchor 정보 저장   : {anchors_path}")
print("\n⚠️ 아직 threshold 보정 전(중간 체크포인트)이라 confidence 임계값이 최적이 아닐 수 있습니다.")
print("   본 학습이 끝난 뒤 노트북의 threshold 탐색 셀을 실행하면 더 정확한 값을 얻습니다.")
