import shutil
from collections import Counter
from pathlib import Path

import torch
from torchvision import datasets, transforms

from config import VAL_DIR, OUTPUT_DIR, MODEL_PATH, IMAGE_SIZE
from model import create_model


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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    class_names = checkpoint["class_names"]

    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def main():
    device = get_device()
    print(f"Kullanılan cihaz: {device}")

    model, model_class_names = load_model(device)

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=get_transform()
    )

    error_dir = OUTPUT_DIR / "misclassified"
    if error_dir.exists():
        shutil.rmtree(error_dir)

    error_dir.mkdir(parents=True, exist_ok=True)

    confusion_counter = Counter()
    total = 0
    wrong = 0

    with torch.no_grad():
        for index in range(len(val_dataset)):
            image_tensor, true_index = val_dataset[index]
            image_path, _ = val_dataset.samples[index]

            image_tensor = image_tensor.unsqueeze(0).to(device)

            outputs = model(image_tensor)
            predicted_index = torch.argmax(outputs, dim=1).item()

            true_class = val_dataset.classes[true_index]
            predicted_class = model_class_names[predicted_index]

            total += 1

            if true_class != predicted_class:
                wrong += 1

                confusion_counter[(true_class, predicted_class)] += 1

                target_folder = error_dir / f"{true_class}_as_{predicted_class}"
                target_folder.mkdir(parents=True, exist_ok=True)

                source_path = Path(image_path)
                target_path = target_folder / source_path.name

                shutil.copy2(source_path, target_path)

    print("\nHATA ÖZETİ")
    print("-" * 50)
    print(f"Toplam validation görseli: {total}")
    print(f"Yanlış tahmin sayısı:      {wrong}")
    print(f"Doğru tahmin sayısı:       {total - wrong}")

    print("\nEn çok karışan sınıflar:")
    print("-" * 50)

    for (true_class, predicted_class), count in confusion_counter.most_common():
        print(f"{true_class:<20} -> {predicted_class:<20} : {count}")

    print(f"\nHatalı görseller kaydedildi: {error_dir}")


if __name__ == "__main__":
    main()