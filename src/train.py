import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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

from model import create_model


def get_device():
    """
    Eğitim için uygun cihazı seçer.
    Mac'te MPS varsa onu kullanır.
    NVIDIA GPU varsa CUDA kullanır.
    Yoksa CPU kullanır.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_transforms():
    """
    Train tarafında augmentation uygulanır.
    Validation tarafında augmentation uygulanmaz.
    """

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def calculate_class_weights(dataset, num_classes):
    """
    Sınıf dengesizliğini azaltmak için class weight hesaplar.

    STATION_WAGON sınıfı diğerlerinden az olduğu için,
    loss fonksiyonunda bu sınıfa biraz daha fazla önem verilmesini sağlar.
    """

    targets = [sample[1] for sample in dataset.samples]
    class_counts = np.bincount(targets, minlength=num_classes)

    total_samples = len(targets)
    class_weights = total_samples / (num_classes * class_counts)

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

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


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


def main():
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Kullanılan cihaz: {device}")

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform
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

    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
        "val_acc": []
    }

    best_val_loss = float("inf")
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

        val_loss, val_acc = validate_one_epoch(
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

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                    "model_name": "efficientnet_b0"
                },
                MODEL_PATH
            )

            print("En iyi model kaydedildi.")
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
    print(f"Model kaydedildi: {MODEL_PATH}")
    print(f"Grafikler kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()