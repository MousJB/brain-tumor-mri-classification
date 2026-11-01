import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    """3 conv blocks (Conv-BN-ReLU-MaxPool) + 2 FC layers. Expects (1, 224, 224) inputs."""

    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GrayToRGB(nn.Module):
    """Repeats a 1-channel input 3 times so it fits ImageNet backbones."""

    def forward(self, x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def get_resnet18(num_classes=4, freeze_backbone=True, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # New head, trainable in both modes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return nn.Sequential(GrayToRGB(), model)


def build_model(name, num_classes=4, pretrained=True):
    if name == "simplecnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        return get_resnet18(num_classes, freeze_backbone=True, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")
