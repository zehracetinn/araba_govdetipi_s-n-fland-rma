import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    TRAIN_DIR,
    VAL_DIR,
    OUTPUT_DIR,
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)

from model import create_model, MODEL_NAME
from logging_utils import setup_logging
from transforms_utils import CenterLetterbox, RandomLetterbox


def pil_loader_rgb(path):
    """
    Tüm görselleri RGB formatına çevirir.
    Böylece PNG/transparency kaynaklı PIL uyarıları ve kanal sorunları azalır.
    """
    with open(path, "rb") as file:
        image = Image.open(file)
        return image.convert("RGB")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_transforms():
    """
    Train tarafında augmentation uygulanır.
    Validation tarafında inference (arayüz/PredictionScript) ile birebir
    aynı sade preprocessing yapılır.

    ÖNEMLİ - genelleme düzeltmesi:
    Veri setinde sedanlar hep geniş çerçeveli (en-boy ~2.4), wagonlar
    kareye yakın (~1.6). Bu yüzden eski model "geniş çerçeve = sedan"
    gibi sahte bir kısayol öğrenip gerçek fotoğraflarda sedanları wagon
    sanıyordu. Bunu kırmak için:
      - RandomResizedCrop ratio aralığı çok geniş tutuldu: (0.5, 2.0).
        Böylece model aynı aracı hem geniş hem kare çerçevede görür ve
        çerçeve oranını sınıf ipucu olarak kullanamaz.
      - Inference tarafında CenterCrop YERİNE doğrudan (224x224) resize
        kullanıyoruz; böylece aracın ön/arka (bagaj) kısmı kesilmez ve
        tüm araç modele gösterilir.
    """

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25
        ),
        # Aracın tamamını (arka dahil) koruyarak rastgele boyut/konumda yerleştir.
        # Crop YOK -> sedan/wagon ve suv/pickup ayrımı için arka kısım hep görünür.
        RandomLetterbox(IMAGE_SIZE, min_scale=0.6, max_scale=1.0),
        transforms.RandomRotation(degrees=8, fill=124),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        CenterLetterbox(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def calculate_class_weights(dataset, num_classes):
    targets = [sample[1] for sample in dataset.samples]
    class_counts = np.bincount(targets, minlength=num_classes)

    total_samples = len(targets)

    class_weights = []
    for count in class_counts:
        if count == 0:
            class_weights.append(0.0)
        else:
            class_weights.append(total_samples / (num_classes * count))

    return torch.tensor(class_weights, dtype=torch.float32), class_counts


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, dim=1)

        running_loss += loss.item() * images.size(0)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predictions = torch.max(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_accuracy, macro_f1


def save_training_curves(history):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["val_macro_f1"], label="Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1-Score")
    plt.title("Validation Macro F1-Score")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "macro_f1_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    start_time = time.time()

    setup_logging("train")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Kullanılan cihaz: {device}")

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
        loader=pil_loader_rgb
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform,
        loader=pil_loader_rgb
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("\nSınıflar:")
    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    class_weights, class_counts = calculate_class_weights(
        train_dataset,
        num_classes
    )

    print("\nTrain sınıf dağılımı:")
    for class_name, count in zip(class_names, class_counts):
        print(f"{class_name:<20} {count}")

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = create_model(num_classes=num_classes)
    model = model.to(device)

    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    best_macro_f1 = 0.0
    patience = 5
    patience_counter = 0

    print("\nEğitim başlıyor...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, val_macro_f1 = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Val Macro F1-Score: {val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                    "model_name": MODEL_NAME,
                    "best_macro_f1": best_macro_f1
                },
                MODEL_PATH
            )

            print("En iyi Macro F1 model kaydedildi.")
        else:
            patience_counter += 1
            print(f"EarlyStopping sayacı: {patience_counter}/{patience}")

        print()

        if patience_counter >= patience:
            print("EarlyStopping çalıştı. Eğitim durduruldu.")
            break

    save_training_curves(history)

    history_path = OUTPUT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)

    elapsed_time = time.time() - start_time
    print(f"Eğitim tamamlandı. Süre: {elapsed_time / 60:.2f} dakika")
    print(f"En iyi Macro F1: {best_macro_f1:.4f}")
    print(f"Model kaydedildi: {MODEL_PATH}")
    print(f"Grafikler kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()