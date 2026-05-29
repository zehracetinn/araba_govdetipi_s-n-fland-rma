"""
Teşhis aracı: gerçek (görülmemiş) görsellerde modelin neden yanıldığını anlamak için.

Bir klasördeki her görsel için, FARKLI preprocessing stratejileri altında
modelin tam olasılık dağılımını yazdırır. Böylece sorunun preprocessing
kaynaklı (center-crop aracı kesiyor) mu yoksa gerçek bir model zayıflığı mı
olduğunu görebiliriz.

Kullanım:
    cd src && ../venv/bin/python diagnose.py ../debug_images
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import MODEL_PATH, IMAGE_SIZE, DISPLAY_NAMES
from model import create_model


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# İki farklı preprocessing stratejisi:
PREPROCESS_VARIANTS = {
    # Mevcut arayüzdeki yöntem: kısa kenarı 256 yap, ortadan 224 kırp.
    # Geniş fotoğraflarda aracın ön/arka kısmını kesebilir.
    "center_crop": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Tüm görseli 224x224'e sıkıştırır (en-boy oranı bozulur ama araç tamamen görünür).
    "squash_resize": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def predict_with(transform, image, model, class_names, device):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
    order = probs.argsort()[::-1]
    top = [(class_names[i], float(probs[i])) for i in order[:3]]
    return top


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../debug_images")

    images = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    if not images:
        print(f"Görsel bulunamadı: {folder}")
        print("Başarısız olan görsellerini debug_images/ klasörüne kopyala.")
        return

    device = get_device()
    model, class_names = load_model(device)

    print(f"Cihaz: {device} | {len(images)} görsel | Klasör: {folder}\n")

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        print(f"### {image_path.name}  (orijinal boyut: {image.size[0]}x{image.size[1]})")

        for variant_name, transform in PREPROCESS_VARIANTS.items():
            top = predict_with(transform, image, model, class_names, device)
            top_str = "   ".join(
                f"{DISPLAY_NAMES.get(n, n)}: %{p * 100:.1f}" for n, p in top
            )
            print(f"  [{variant_name:<13}] {top_str}")
        print()


if __name__ == "__main__":
    main()
