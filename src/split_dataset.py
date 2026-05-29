import random
import shutil
from pathlib import Path

from PIL import Image

from config import (
    DATA_SOURCE_DIR,
    FOLDER_TO_CLASS,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    VAL_RATIO,
    TEST_RATIO,
)
from logging_utils import setup_logging


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RANDOM_SEED = 42

# Her sınıftan maksimum kaç görsel kullanılacak?
# Sınıflar arası dengeyi korumak için büyük sınıfları kırpıyoruz.
MAX_IMAGES_PER_CLASS = 700

# Bu sayının altındaki sınıflarda uyarı vereceğiz
MIN_IMAGES_WARNING = 250


def clear_directory(directory: Path):
    """
    Train/val/test klasörlerini temizler.
    Böylece her split işleminde eski dosyalar karışmaz ve
    tekrar çalıştırıldığında veri sızıntısı (data leakage) olmaz.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def is_image_valid(image_path: Path) -> bool:
    """
    Bozuk görselleri eğitime sokmamak için dosyayı doğrular.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def copy_images(image_paths, target_dir: Path):
    """
    Görselleri hedef klasöre kopyalar.
    İsim çakışmasını önlemek için başına index ekler.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths):
        target_name = f"{index:05d}_{image_path.name}"
        target_path = target_dir / target_name
        shutil.copy2(image_path, target_path)


def split_class_images(folder_name: str, class_name: str):
    source_class_dir = DATA_SOURCE_DIR / folder_name

    if not source_class_dir.exists():
        print(f"{class_name:<20} kaynak klasörü bulunamadı: {source_class_dir}")
        return

    image_paths = [
        file for file in source_class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
        and is_image_valid(file)
    ]

    random.shuffle(image_paths)

    raw_count = len(image_paths)

    if raw_count == 0:
        print(f"{class_name:<20} Raw: 0     UYARI: boş sınıf")
        return

    # Çok büyük sınıflardan dengeyi korumak için kırpıyoruz.
    selected_images = image_paths[:MAX_IMAGES_PER_CLASS]
    selected_count = len(selected_images)

    # Önce test, sonra val, kalanı train olacak şekilde böl.
    # Aynı görsel birden fazla sete giremez (kesin ayrım => leakage yok).
    test_count = int(selected_count * TEST_RATIO)
    val_count = int(selected_count * VAL_RATIO)

    test_images = selected_images[:test_count]
    val_images = selected_images[test_count:test_count + val_count]
    train_images = selected_images[test_count + val_count:]

    copy_images(train_images, TRAIN_DIR / class_name)
    copy_images(val_images, VAL_DIR / class_name)
    copy_images(test_images, TEST_DIR / class_name)

    warning_text = ""
    if selected_count < MIN_IMAGES_WARNING:
        warning_text = "UYARI: az veri"

    print(
        f"{class_name:<20} "
        f"Raw: {raw_count:<6} "
        f"Kullanılan: {selected_count:<5} "
        f"Train: {len(train_images):<5} "
        f"Val: {len(val_images):<5} "
        f"Test: {len(test_images):<5} "
        f"{warning_text}"
    )


def main():
    setup_logging("split")
    random.seed(RANDOM_SEED)

    if not DATA_SOURCE_DIR.exists():
        print(f"Veri kaynağı bulunamadı: {DATA_SOURCE_DIR}")
        return

    clear_directory(TRAIN_DIR)
    clear_directory(VAL_DIR)
    clear_directory(TEST_DIR)

    train_ratio = int((1 - VAL_RATIO - TEST_RATIO) * 100)
    print("Train/Val/Test split işlemi başlıyor...")
    print(f"Oranlar -> Train: %{train_ratio}  "
          f"Val: %{int(VAL_RATIO * 100)}  Test: %{int(TEST_RATIO * 100)}\n")

    for folder_name, class_name in FOLDER_TO_CLASS.items():
        split_class_images(folder_name, class_name)

    print("\nSplit işlemi tamamlandı.")
    print(f"Train klasörü: {TRAIN_DIR}")
    print(f"Val klasörü:   {VAL_DIR}")
    print(f"Test klasörü:  {TEST_DIR}")


if __name__ == "__main__":
    main()
