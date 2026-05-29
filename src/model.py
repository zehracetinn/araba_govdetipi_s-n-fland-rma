import torch.nn as nn
from torchvision import models


# Kaydedilen modelin checkpoint'inde tutulan mimari adı.
# evaluate / predict aşamalarında doğru mimariyi kurmak için kullanılır.
MODEL_NAME = "mobilenet_v3_large"


def create_model(num_classes: int):
    """
    MobileNetV3-Large tabanlı transfer learning modeli oluşturur.

    Neden MobileNetV3-Large?
    - Hafif mimari: kaydedilen model ~17 MB, yani 95 MB sınırının çok altında.
    - Çıkarım (inference) hızı yüksek -> web arayüzünde hızlı tahmin.
    - ImageNet ön eğitimi sayesinde küçük veri setinde dahi güçlü genelleme.

    ImageNet üzerinde önceden eğitilmiş ağırlıklar alınır ve son
    sınıflandırma katmanı 8 araba gövde tipi sınıfımıza göre değiştirilir.
    """

    weights = models.MobileNet_V3_Large_Weights.DEFAULT

    model = models.mobilenet_v3_large(weights=weights)

    # MobileNetV3-Large classifier yapısı:
    #   [Linear(960->1280), Hardswish, Dropout(0.2), Linear(1280->1000)]
    # Son Linear katmanın giriş boyutunu (1280) alıp kendi başlığımızı kuruyoruz.
    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model
