import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
    deeplabv3_mobilenet_v3_large,
)


class MobileNetV3UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.classifier.low_classifier  = nn.Conv2d(40,  1, kernel_size=1)
        self.model.classifier.high_classifier = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        return {"out": self.model(x)["out"]}


def build_mobileunet(device: torch.device) -> MobileNetV3UNet:
    model = MobileNetV3UNet().to(device)
    print(f"MobileNetV3-UNet (LR-ASPP) initialised on: {device}")
    return model


class FPN(nn.Module):
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


def build_fpn(device: torch.device) -> FPN:
    model = FPN().to(device)
    print(f"FPN (ResNet34, smp) initialised on: {device}")
    return model


class SAM2SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        in_ch = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_ch, 1, kernel_size=1)

        if self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(aux_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        return {"out": self.model(x)["out"]}


def build_sam2(device: torch.device) -> SAM2SegmentationModel:
    model = SAM2SegmentationModel().to(device)
    print(f"SAM2 (DeepLabV3 + MobileNetV3-Large) initialised on: {device}")
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
