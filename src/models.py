import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet34_Weights,
    mobilenet_v3_large,
    resnet34,
)
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MobileNetV3UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
        self.decode16 = ConvBlock(960 + 80, 256)
        self.decode32 = ConvBlock(256 + 40, 128)
        self.decode64 = ConvBlock(128 + 24, 64)
        self.decode128 = ConvBlock(64 + 16, 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        input_size = x.shape[-2:]
        features = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in (0, 2, 4, 7):
                features.append(x)

        skip128, skip64, skip32, skip16 = features

        x = F.interpolate(x, size=skip16.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode16(torch.cat((x, skip16), dim=1))
        x = F.interpolate(x, size=skip32.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode32(torch.cat((x, skip32), dim=1))
        x = F.interpolate(x, size=skip64.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode64(torch.cat((x, skip64), dim=1))
        x = F.interpolate(x, size=skip128.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode128(torch.cat((x, skip128), dim=1))
        x = self.output(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return {"out": x}


def build_mobileunet(device: torch.device) -> MobileNetV3UNet:
    model = MobileNetV3UNet().to(device)
    print(f"MobileNetV3-UNet initialised on: {device}")
    return model


class ConvRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PyTorchFPN(nn.Module):
    def __init__(self, num_classes: int = 1, fpn_channels: int = 128):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.conv1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.lat4 = nn.Conv2d(512, fpn_channels, 1)
        self.lat3 = nn.Conv2d(256, fpn_channels, 1)
        self.lat2 = nn.Conv2d(128, fpn_channels, 1)
        self.lat1 = nn.Conv2d(64, fpn_channels, 1)

        self.smooth4 = ConvRelu(fpn_channels, fpn_channels)
        self.smooth3 = ConvRelu(fpn_channels, fpn_channels)
        self.smooth2 = ConvRelu(fpn_channels, fpn_channels)
        self.smooth1 = ConvRelu(fpn_channels, fpn_channels)

        self.seg_head = nn.Sequential(
            ConvRelu(fpn_channels * 4, fpn_channels),
            nn.Dropout2d(0.2),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        inp_size = x.shape[-2:]
        c1 = self.layer1(self.conv1(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        target = p1.shape[-2:]
        combined = torch.cat([
            p1,
            F.interpolate(p2, size=target, mode="bilinear", align_corners=False),
            F.interpolate(p3, size=target, mode="bilinear", align_corners=False),
            F.interpolate(p4, size=target, mode="bilinear", align_corners=False),
        ], dim=1)

        out = self.seg_head(combined)
        out = F.interpolate(out, size=inp_size, mode="bilinear", align_corners=False)
        return {"out": out}


class FPNWrapper(nn.Module):
    def __init__(self):
        import segmentation_models_pytorch as smp
        super().__init__()
        self.model = smp.FPN(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> dict:
        return {"out": self.model(x)}


def build_fpn(device: torch.device):
    try:
        model = FPNWrapper().to(device)
        print("Using segmentation_models_pytorch FPN with ResNet34 backbone.")
    except Exception as exc:
        print(f"SMP unavailable ({exc}). Falling back to native PyTorchFPN.")
        model = PyTorchFPN().to(device)
    print(f"FPN initialised on: {device}")
    return model


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q2, _ = self.self_attn(q, q, q)
        q = self.norm1(q + q2)
        q2, _ = self.cross_attn(q, kv, kv)
        q = self.norm2(q + q2)
        q = self.norm3(q + self.mlp(q))
        return q


class SAM2SegmentationModel(nn.Module):
    def __init__(self, in_channels: int = 3, d_model: int = 128):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        self.prompt_tokens = nn.Parameter(torch.randn(1, 4, d_model))

        self.attn_block1 = TwoWayAttentionBlock(d_model=d_model, nhead=4)
        self.attn_block2 = TwoWayAttentionBlock(d_model=d_model, nhead=4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]

        f = self.stage2(self.stage1(self.stem(x)))

        feat_flat = f.flatten(2).permute(0, 2, 1)

        prompts = self.prompt_tokens.expand(B, -1, -1)
        prompts = self.attn_block1(prompts, feat_flat)
        prompts = self.attn_block2(prompts, feat_flat)

        ctx = prompts.mean(dim=1, keepdim=True).permute(0, 2, 1).unsqueeze(-1)
        f_modulated = f + ctx

        mask_logits = self.decoder(f_modulated)
        mask_logits = F.interpolate(
            mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"out": mask_logits}


def build_sam2(device: torch.device) -> SAM2SegmentationModel:
    model = SAM2SegmentationModel().to(device)
    print(f"SAM 2 Segmentation Model initialised on: {device}")
    return model


class FCNBinary(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
        self.fcn = fcn_resnet50(weights=weights)

        in_channels = self.fcn.classifier[4].in_channels
        self.fcn.classifier[4] = nn.Conv2d(in_channels, 1, kernel_size=1)

        if self.fcn.aux_classifier is not None:
            aux_in = self.fcn.aux_classifier[4].in_channels
            self.fcn.aux_classifier[4] = nn.Conv2d(aux_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        return {"out": self.fcn(x)["out"]}


def build_fcn(device: torch.device) -> FCNBinary:
    model = FCNBinary(pretrained=True).to(device)
    print(f"FCN (ResNet50) initialised on: {device}")
    return model
