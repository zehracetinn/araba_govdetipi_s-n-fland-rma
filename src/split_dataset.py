import random
import shutil
from pathlib import Path

from config import RAW_DIR, TRAIN_DIR, VAL_DIR, CLASS_NAMES


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VAL_RATIO = 0.20
RANDOM_SEED = 42


def clear_directory(directory: Path):
    """
    Train/val klasörlerini temizler.
    Böylece split işlemi her çalıştığında eski dosyalar karışmaz.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def copy_images(image_paths, target_dir: Path):
    """
    Görselleri hedef klasöre kopyalar.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        target_path = target_dir / image_path.name
        shutil.copy2(image_path, target_path)


def split_class_images(class_name: str):
    """
    Bir sınıfa ait raw görselleri train ve validation olarak ayırır.
    """
    raw_class_dir = RAW_DIR / class_name

    if not raw_class_dir.exists():
        print(f"{class_name}: raw klasörü bulunamadı.")
        return

    image_paths = [
        file for file in raw_class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]

    random.shuffle(image_paths)

    total_count = len(image_paths)

    if total_count == 0:
        print(f"{class_name}: 0 görsel bulundu.")
        return

    val_count = int(total_count * VAL_RATIO)

    val_images = image_paths[:val_count]
    train_images = image_paths[val_count:]

    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name

    copy_images(train_images, train_class_dir)
    copy_images(val_images, val_class_dir)

    print(
        f"{class_name:<20} Toplam: {total_count:<5} "
        f"Train: {len(train_images):<5} Val: {len(val_images):<5}"
    )


def main():
    random.seed(RANDOM_SEED)

    clear_directory(TRAIN_DIR)
    clear_directory(VAL_DIR)

    print("Dataset train/validation olarak ayrılıyor...\n")

    for class_name in CLASS_NAMES:
        split_class_images(class_name)

    print("\nSplit işlemi tamamlandı.")


if __name__ == "__main__":
    main()