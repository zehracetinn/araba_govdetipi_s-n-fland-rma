import os

classes = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "OPEN_WHEEL",
    "SEDAN",
    "HATCHBACK",
    "PICK_UP"
]

folders = [
    "dataset/train",
    "dataset/val",
    "src",
    "app",
    "outputs",
    "notebooks",
    "report"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for split in ["train", "val"]:
    for class_name in classes:
        os.makedirs(f"dataset/{split}/{class_name}", exist_ok=True)

print("Proje klasör yapısı başarıyla oluşturuldu.")