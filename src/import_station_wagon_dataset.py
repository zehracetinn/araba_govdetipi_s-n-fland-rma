import shutil
from pathlib import Path

from config import RAW_DIR


SOURCE_DIRS = [
    Path("downloads/car_data"),
    Path("downloads/stanford_classes"),
    Path("car_data"),
]

TARGET_DIR = RAW_DIR / "STATION_WAGON"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Station wagon için güvenli kelimeler
INCLUDE_KEYWORDS = [
    "wagon",
    "estate",
    "avant",
    "variant",
    "combi",
    "sportwagon",
    "sports tourer"
]

# Bunlar varsa alma, çünkü sınıfı bozabilir
EXCLUDE_KEYWORDS = [
    "hatchback",
    "van",
    "suv",
    "truck",
    "pickup",
    "sedan",
    "coupe",
    "convertible"
]


def is_valid_station_wagon_folder(folder_path: Path) -> bool:
    folder_name = folder_path.name.lower()

    has_include = any(keyword in folder_name for keyword in INCLUDE_KEYWORDS)
    has_exclude = any(keyword in folder_name for keyword in EXCLUDE_KEYWORDS)

    return has_include and not has_exclude


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    found_folders = []

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue

        for folder in source_dir.rglob("*"):
            if folder.is_dir() and is_valid_station_wagon_folder(folder):
                found_folders.append(folder)

    if not found_folders:
        print("Temiz station wagon klasörü bulunamadı.")
        return

    print("Aktarılacak temiz station wagon klasörleri:")
    print("-" * 70)

    for folder in found_folders:
        print(folder)

        image_files = [
            file for file in folder.rglob("*")
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        ]

        for image_path in image_files:
            target_name = f"station_wagon_extra_{copied_count:05d}_{image_path.name}"
            target_path = TARGET_DIR / target_name

            if not target_path.exists():
                shutil.copy2(image_path, target_path)
                copied_count += 1

    print("-" * 70)
    print(f"STATION_WAGON sınıfına aktarılan temiz görsel: {copied_count}")
    print(f"Hedef klasör: {TARGET_DIR}")


if __name__ == "__main__":
    main()