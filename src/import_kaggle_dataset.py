import shutil
from pathlib import Path

from config import BASE_DIR, RAW_DIR


SOURCE_DIR = BASE_DIR / "Cars_Body_Type"

# Kaggle klasör adı -> bizim raw klasör adı
CLASS_MAP = {
    "Hatchback": "HATCHBACK",
    "Pick-Up": "PICK_UP",
    "SUV": "SUV",
    "Sedan": "SEDAN",
    "VAN": "VAN",
}

# Şimdilik test klasörünü almıyoruz.
# Çünkü test verisini final kontrol için ayrı tutabiliriz.
SPLITS_TO_IMPORT = ["train", "valid"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def copy_class_images(split_name: str, source_class: str, target_class: str):
    source_class_dir = SOURCE_DIR / split_name / source_class
    target_class_dir = RAW_DIR / target_class

    if not source_class_dir.exists():
        print(f"Bulunamadı: {source_class_dir}")
        return 0

    target_class_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        file for file in source_class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]

    copied_count = 0

    for index, image_path in enumerate(image_files, start=1):
        # Aynı dosya adı çakışmasın diye başına split ve sınıf adı ekliyoruz
        new_file_name = f"{split_name}_{source_class}_{index:05d}{image_path.suffix.lower()}"
        target_path = target_class_dir / new_file_name

        shutil.copy2(image_path, target_path)
        copied_count += 1

    return copied_count


def main():
    if not SOURCE_DIR.exists():
        print(f"Kaggle klasörü bulunamadı: {SOURCE_DIR}")
        print("Cars_Body_Type klasörünü yazlab3 içine koymalısın.")
        return

    print("Kaggle veri seti raw klasörüne aktarılıyor...\n")

    total = 0

    for split_name in SPLITS_TO_IMPORT:
        print(f"{split_name.upper()} klasörü işleniyor...")

        for source_class, target_class in CLASS_MAP.items():
            copied = copy_class_images(split_name, source_class, target_class)
            total += copied

            print(f"{source_class:<12} -> {target_class:<15} {copied} görsel")

        print()

    print(f"Toplam aktarılan görsel: {total}")
    print("Aktarım tamamlandı.")


if __name__ == "__main__":
    main()