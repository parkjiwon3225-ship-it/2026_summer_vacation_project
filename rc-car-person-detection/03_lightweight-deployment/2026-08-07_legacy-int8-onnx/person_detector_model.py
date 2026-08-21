import torch
import torch.nn as nn


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
            nn.ReLU6(inplace=True),
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

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

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
            k=3,
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

        refine_layers = []

        prev = dw2_c

        for c in refine_c:
            refine_layers.append(
                DepthwiseSeparable(
                    prev,
                    c,
                    stride=1
                )
            )

            prev = c


        self.refine = nn.Sequential(
            *refine_layers
        )

        self.out_channels = prev


    def forward(self, x):

        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.refine(x)

        return x



class MultiAnchorHead(nn.Module):

    def __init__(
        self,
        in_channels,
        num_anchors=5
    ):
        super().__init__()

        self.num_anchors = num_anchors

        self.refine = DepthwiseSeparable(
            in_channels,
            in_channels
        )

        self.predict = nn.Conv2d(
            in_channels,
            num_anchors * 5,
            kernel_size=1
        )


    def forward(self,x):

        x = self.refine(x)

        out = self.predict(x)

        b,_,h,w = out.shape

        out = out.view(
            b,
            self.num_anchors,
            5,
            h,
            w
        )

        return out



class PersonDetectorV2(nn.Module):

    def __init__(
        self,
        backbone_channels=(32,64,128,128,256,256,256),
        num_anchors=5
    ):

        super().__init__()

        self.backbone = Backbone(
            backbone_channels
        )

        self.head = MultiAnchorHead(
            self.backbone.out_channels,
            num_anchors
        )


    def forward(self,x):

        features = self.backbone(x)

        out = self.head(features)

        return out