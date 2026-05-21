from pathlib import Path
from PIL import Image
from config import TRAIN_DIR, VAL_DIR, CLASS_NAMES


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image_valid(image_path: Path) -> bool:
    """
    Görsel dosyası gerçekten okunabiliyor mu kontrol eder.
    Bozuk görseller eğitim sırasında hata çıkarabilir.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def count_images(split_dir: Path, split_name: str):
    """
    Train veya validation klasöründeki sınıfların görsel sayılarını yazdırır.
    """
    print(f"\n{split_name.upper()} SETİ")
    print("-" * 40)

    total_count = 0
    broken_images = []

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name

        if not class_dir.exists():
            print(f"{class_name:<20} klasör bulunamadı!")
            continue

        image_files = [
            file for file in class_dir.iterdir()
            if file.suffix.lower() in VALID_EXTENSIONS
        ]

        valid_count = 0

        for image_path in image_files:
            if is_image_valid(image_path):
                valid_count += 1
            else:
                broken_images.append(image_path)

        total_count += valid_count

        print(f"{class_name:<20} {valid_count:>5} görsel")

    print("-" * 40)
    print(f"Toplam: {total_count} görsel")

    if broken_images:
        print("\nBozuk görseller:")
        for img in broken_images:
            print(img)
    else:
        print("Bozuk görsel bulunmadı.")


def main():
    count_images(TRAIN_DIR, "train")
    count_images(VAL_DIR, "validation")


if __name__ == "__main__":
    main()