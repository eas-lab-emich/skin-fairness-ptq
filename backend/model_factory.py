"""
model_factory.py
================
Shared model builder for all architectures used in the skin fairness study.

Supported architectures:
  - resnet18          torchvision ResNet-18 (224x224, ImageNet head)
  - resnet34          torchvision ResNet-34 (224x224, ImageNet head)
  - resnet50          torchvision ResNet-50 (224x224, ImageNet head)
  - mobilenet_v2      torchvision MobileNetV2 (224x224)
  - shufflenet_v2_x1_0  torchvision ShuffleNetV2-1.0x (224x224)
  - vgg11             torchvision VGG-11 (224x224)
"""

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "vgg11",
]


def build_skin_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a model for skin classification on Fitzpatrick17k.

    All architectures produce a model whose final classifier outputs `num_classes` logits
    and accepts 224x224 RGB images.

    Args:
        arch: Architecture name. Must be one of SUPPORTED_ARCHS.
        num_classes: Number of output classes.
        pretrained: If True, initialise torchvision models with ImageNet weights
                    (not available for CIFAR ResNets).

    Returns:
        nn.Module ready for training / evaluation (not yet moved to device).
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported arch: {arch!r}. Choose from: {SUPPORTED_ARCHS}")

    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.resnet18(weights=weights)
        except TypeError:
            model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, int(num_classes))

    elif arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.resnet34(weights=weights)
        except TypeError:
            model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, int(num_classes))

    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.resnet50(weights=weights)
        except TypeError:
            model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, int(num_classes))

    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.mobilenet_v2(weights=weights)
        except TypeError:
            model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, int(num_classes))

    elif arch == "shufflenet_v2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.shufflenet_v2_x1_0(weights=weights)
        except TypeError:
            model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, int(num_classes))

    elif arch == "vgg11":
        weights = models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.vgg11(weights=weights)
        except TypeError:
            model = models.vgg11(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, int(num_classes))

    return model
