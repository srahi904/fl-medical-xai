"""
model.py
--------
Defines the CNN classifier used by every FL client and the global server model.
Uses a pretrained ResNet18 backbone (transfer learning) with the final layer
replaced for our number of medical classes.

Kept as a single, simple architecture on purpose: in Federated Learning every
client and the server must use the EXACT same architecture so that weights
can be averaged (FedAvg) layer by layer.
"""

import torch
import torch.nn as nn
from torchvision import models


class MedicalCNN(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Keep everything except the final FC layer -> we use this as our
        # feature extractor. Grad-CAM will hook into `self.features[-1]`
        # (the last conv block) later.
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # up to last conv block
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)          # [B, 512, H, W] conv feature maps
        pooled = self.pool(x)
        pooled = torch.flatten(pooled, 1)
        out = self.classifier(pooled)
        return out


def build_model(num_classes: int = 2, pretrained: bool = True, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalCNN(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)


def get_model_parameters(model: nn.Module):
    """Extract model weights as a list of numpy arrays (needed by Flower)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters):
    """Load a list of numpy arrays (from Flower server) back into the model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model
