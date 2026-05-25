import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from torchvision import datasets, transforms

from config import (
    VAL_DIR,
    OUTPUT_DIR,
    MODEL_PATH,
    IMAGE_SIZE
)
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

OFFICIAL_CLASS_NAMES = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "OPEN_WHEEL",
    "SEDAN",
    "HATCHBACK",
    "PICK_UP"
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_trained_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def save_normalized_confusion_matrix(y_true, y_pred):
    labels = [1, 2, 3, 4, 5, 6, 7, 8]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        xticklabels=OFFICIAL_CLASS_NAMES,
        yticklabels=OFFICIAL_CLASS_NAMES
    )

    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "normalized_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Normalized confusion matrix kaydedildi: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Kullanılan cihaz: {device}")

    model, model_class_names = load_trained_model(device)

    print("\nModel class sırası:")
    for index, class_name in enumerate(model_class_names):
        print(f"{index}: {class_name}")

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=get_val_transform()
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for image_tensor, true_class_index in val_dataset:
            image_tensor = image_tensor.unsqueeze(0).to(device)

            outputs = model(image_tensor)
            predicted_index = torch.argmax(outputs, dim=1).item()

            true_class_name = val_dataset.classes[true_class_index]
            predicted_class_name = model_class_names[predicted_index]

            true_label = OFFICIAL_LABEL_MAP[true_class_name]
            predicted_label = OFFICIAL_LABEL_MAP[predicted_class_name]

            y_true.append(true_label)
            y_pred.append(predicted_label)

    labels = [1, 2, 3, 4, 5, 6, 7, 8]

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    precision_weighted = precision_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    f1_weighted = f1_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    print("\nGENEL METRİKLER")
    print("-" * 40)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Macro Recall:       {recall_macro:.4f}")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall:    {recall_weighted:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 40)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=OFFICIAL_CLASS_NAMES,
        zero_division=0
    )

    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=OFFICIAL_CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )

    report_path = OUTPUT_DIR / "classification_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report_dict, file, ensure_ascii=False, indent=4)

    print(f"Classification report kaydedildi: {report_path}")

    save_normalized_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()