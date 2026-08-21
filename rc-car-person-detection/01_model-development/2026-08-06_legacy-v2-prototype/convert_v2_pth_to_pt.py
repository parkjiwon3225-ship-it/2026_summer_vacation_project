import torch
import torch.nn as nn


# ============================================================
# Model Definition
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, groups=1):
        super().__init__()

        padding = k // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                cin,
                cout,
                k,
                stride,
                padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)



class DepthwiseSeparable(nn.Module):

    def __init__(self, cin, cout, stride=1):
        super().__init__()

        self.depthwise = ConvBNAct(
            cin,
            cin,
            k=3,
            stride=stride,
            groups=cin
        )

        self.pointwise = ConvBNAct(
            cin,
            cout,
            k=1
        )


    def forward(self,x):

        x=self.depthwise(x)
        x=self.pointwise(x)

        return x



class Backbone(nn.Module):

    def __init__(
        self,
        channels=(32,64,128,128,256,256,256)
    ):
        super().__init__()

        stem_c, dw1_c, dw2_c, *refine_c = channels


        self.stem = ConvBNAct(
            3,
            stem_c,
            stride=2
        )


        self.down1 = DepthwiseSeparable(
            stem_c,
            dw1_c,
            stride=2
        )


        self.down2 = DepthwiseSeparable(
            dw1_c,
            dw2_c,
            stride=2
        )


        layers=[]

        prev=dw2_c

        for c in refine_c:

            layers.append(
                DepthwiseSeparable(
                    prev,
                    c
                )
            )

            prev=c


        self.refine = nn.Sequential(*layers)

        self.out_channels=prev



    def forward(self,x):

        x=self.stem(x)
        x=self.down1(x)
        x=self.down2(x)
        x=self.refine(x)

        return x



class MultiAnchorHead(nn.Module):

    def __init__(
        self,
        in_channels,
        num_anchors=5
    ):
        super().__init__()

        self.num_anchors=num_anchors


        self.refine=DepthwiseSeparable(
            in_channels,
            in_channels
        )


        self.predict=nn.Conv2d(
            in_channels,
            num_anchors*5,
            kernel_size=1
        )


    def forward(self,x):

        x=self.refine(x)

        x=self.predict(x)

        b,_,h,w=x.shape

        x=x.view(
            b,
            self.num_anchors,
            5,
            h,
            w
        )

        return x



class PersonDetectorV2(nn.Module):

    def __init__(
        self,
        backbone_channels=(32,64,128,128,256,256,256),
        num_anchors=5
    ):
        super().__init__()


        self.backbone=Backbone(
            backbone_channels
        )


        self.head=MultiAnchorHead(
            self.backbone.out_channels,
            num_anchors
        )


    def forward(self,x):

        x=self.backbone(x)

        x=self.head(x)

        return x



# ============================================================
# PTH -> TorchScript PT
# ============================================================

PTH_PATH = "person_detector_v2_best.pth"

PT_PATH = "person_detector_v2_best_script.pt"


device="cpu"


# 모델 생성
model = PersonDetectorV2(
    backbone_channels=(32,64,128,128,256,256,256),
    num_anchors=5
)


# checkpoint 로드

checkpoint=torch.load(
    PTH_PATH,
    map_location=device
)
print(checkpoint.keys())

# 실제 weight만 추출
state_dict = checkpoint["model_state_dict"]

print(list(state_dict.keys())[:10])


model.load_state_dict(
    state_dict
)


model.eval()



# 입력 크기 320x240

dummy_input=torch.randn(
    1,
    3,
    240,
    320
)



# TorchScript 변환

scripted_model=torch.jit.trace(
    model,
    dummy_input
)



# 저장

scripted_model.save(
    PT_PATH
)


print("==============================")
print("변환 완료")
print(PT_PATH)
print("==============================")