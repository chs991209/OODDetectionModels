import torch
import torch.nn as nn
from torchvision import models


def get_animal_model(num_classes=10, pretrained=True):
    """
    Animals-10 분류 및 OOD 탐지를 위한 ResNet18 모델
    """
    # 1. Pretrained ResNet18 로드
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features

    # 2. [핵심] FC Layer 수정 (MC Dropout 적용)
    # 추론 시에도 불확실성(Uncertainty)을 계산할 수 있도록 Dropout 추가
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model