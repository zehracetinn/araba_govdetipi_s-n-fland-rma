import random
import shutil
from pathlib import Path

from config import RAW_DIR, TRAIN_DIR, VAL_DIR, CLASS_NAMES


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

VAL_RATIO = 0.20
RANDOM_SEED = 42

# Her sınıftan maksimum kaç görsel kullanılacak?
MAX_IMAGES_PER_CLASS = 700

# Bu sayının altındaki sınıflarda uyarı vereceğiz
MIN_IMAGES_WARNING = 250


def clear_directory(directory: Path):
    """
    Train/val klasörlerini temizler.
    Böylece her split işleminde eski dosyalar karışmaz.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


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


def split_class_images(class_name: str):
    raw_class_dir = RAW_DIR / class_name

    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    if not raw_class_dir.exists():
        print(f"{class_name:<20} raw klasörü bulunamadı.")
        return

    image_paths = [
        file for file in raw_class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]

    random.shuffle(image_paths)

    raw_count = len(image_paths)

    if raw_count == 0:
        print(f"{class_name:<20} Raw: 0     Kullanılan: 0     Train: 0     Val: 0     UYARI: boş sınıf")
        return

    # Çok büyük sınıflardan sadece 500 görsel seçiyoruz.
    selected_images = image_paths[:MAX_IMAGES_PER_CLASS]

    selected_count = len(selected_images)
    val_count = int(selected_count * VAL_RATIO)

    val_images = selected_images[:val_count]
    train_images = selected_images[val_count:]

    copy_images(train_images, train_class_dir)
    copy_images(val_images, val_class_dir)

    warning_text = ""

    if selected_count < MIN_IMAGES_WARNING:
        warning_text = "UYARI: az veri"

    print(
        f"{class_name:<20} "
        f"Raw: {raw_count:<6} "
        f"Kullanılan: {selected_count:<5} "
        f"Train: {len(train_images):<5} "
        f"Val: {len(val_images):<5} "
        f"{warning_text}"
    )


def main():
    random.seed(RANDOM_SEED)

    clear_directory(TRAIN_DIR)
    clear_directory(VAL_DIR)

    print("Dengeli dataset split işlemi başlıyor...\n")

    for class_name in CLASS_NAMES:
        split_class_images(class_name)

    print("\nSplit işlemi tamamlandı.")


if __name__ == "__main__":
    main()