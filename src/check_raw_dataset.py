
from pathlib import Path
from PIL import Image

from config import RAW_DIR, CLASS_NAMES


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image_valid(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    print("\nRAW DATASET KONTROLÜ")
    print("-" * 45)

    total_count = 0
    broken_images = []

    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            print(f"{class_name:<20} klasör yok")
            continue

        image_files = [
            file for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        ]

        valid_count = 0

        for image_path in image_files:
            if is_image_valid(image_path):
                valid_count += 1
            else:
                broken_images.append(image_path)

        total_count += valid_count
        print(f"{class_name:<20} {valid_count:>5} görsel")

    print("-" * 45)
    print(f"Toplam raw görsel: {total_count}")

    if broken_images:
        print("\nBozuk görseller:")
        for image in broken_images:
            print(image)
    else:
        print("Bozuk görsel bulunmadı.")


if __name__ == "__main__":
    main()