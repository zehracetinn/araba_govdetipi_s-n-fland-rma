from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMAGE_SIZE = 224
MODEL_PATH = Path("outputs/best_model.pth")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


OFFICIAL_LABEL_MAP = {
    "SUV": 1,
    "VAN": 2,
    "STATION_WAGON": 3,
    "MICRO": 4,
    "OPEN_WHEEL": 5,
    "SEDAN": 6,
    "HATCHBACK": 7,
    "PICK_UP": 8
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_model(num_classes):
    """
    Eğitimde kullandığımız EfficientNet-B0 mimarisinin aynısı.
    Burada weights=None kullanıyoruz çünkü eğitilmiş ağırlıkları best_model.pth içinden yüklüyoruz.
    """

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def get_transform():
    """
    Test/predict aşamasında random augmentation yapılmaz.
    Validation ile aynı preprocessing uygulanır.
    """

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model():
    device = get_device()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    class_names = checkpoint["class_names"]

    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, class_names, device


def predict_single_image(image_path, model, class_names, device, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_index = torch.argmax(outputs, dim=1).item()

    predicted_class_name = class_names[predicted_index]
    predicted_label = OFFICIAL_LABEL_MAP[predicted_class_name]

    return predicted_label


def get_image_files(folder_path):
    folder_path = Path(folder_path)

    image_files = [
        file for file in folder_path.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]

    return sorted(image_files)


def write_predictions(lines):
    """
    Büyük/küçük harf sorunu yaşamamak için hem Preds.txt hem preds.txt üretiyoruz.
    Colab ortamında /content varsa oraya da yazar.
    """

    output_paths = [
        Path("Preds.txt"),
        Path("preds.txt")
    ]

    if Path("/content").exists():
        output_paths.extend([
            Path("/content/Preds.txt"),
            Path("/content/preds.txt")
        ])

    for output_path in output_paths:
        with open(output_path, "w", encoding="utf-8") as file:
            for line in lines:
                file.write(line + "\n")

    print("Tahmin dosyaları oluşturuldu:")
    for output_path in output_paths:
        print(output_path)


def Predict(file_path):
    """
    Hocanın test scriptinin çağıracağı ana fonksiyon.

    Parametre:
        file_path: test görsellerinin bulunduğu klasör yolu

    Çıktı formatı:
        image_name.jpg | Pred: 5
    """

    model, class_names, device = load_model()
    transform = get_transform()

    image_files = get_image_files(file_path)

    prediction_lines = []

    for image_path in image_files:
        predicted_label = predict_single_image(
            image_path=image_path,
            model=model,
            class_names=class_names,
            device=device,
            transform=transform
        )

        line = f"{image_path.name} | Pred: {predicted_label}"
        prediction_lines.append(line)

    write_predictions(prediction_lines)

    return prediction_lines


if __name__ == "__main__":
    # Lokal test için örnek:
    # testdata klasörü varsa çalıştırır.
    test_folder = Path("testdata")

    if test_folder.exists():
        Predict(test_folder)
    else:
        print("testdata klasörü bulunamadı.")
        print("Kullanım örneği: Predict('testdata')")