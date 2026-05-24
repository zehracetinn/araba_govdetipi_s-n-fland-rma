import torch.nn as nn
from torchvision import models


def create_model(num_classes: int):
    """
    EfficientNet-B0 tabanlı transfer learning modeli oluşturur.

    ImageNet üzerinde önceden eğitilmiş EfficientNet-B0 alınır.
    Son sınıflandırma katmanı bizim 8 araba gövde tipi sınıfımıza göre değiştirilir.
    """

    weights = models.EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model