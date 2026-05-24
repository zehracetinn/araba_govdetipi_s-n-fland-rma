import shutil
from pathlib import Path
from PIL import Image

from config import RAW_DIR


SOURCE_DIR = Path("downloads/formula_one_cars")
TARGET_DIR = RAW_DIR / "OPEN_WHEEL"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image_valid(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    if not SOURCE_DIR.exists():
        print(f"Kaynak klasör bulunamadı: {SOURCE_DIR}")
        return

    # Eski OPEN_WHEEL verisini temizle
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        file for file in SOURCE_DIR.rglob("*")
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]

    copied_count = 0
    skipped_broken = 0

    for image_path in image_files:
        if not is_image_valid(image_path):
            print(f"Bozuk görsel atlandı: {image_path}")
            skipped_broken += 1
            continue

        target_name = f"open_wheel_{copied_count:05d}_{image_path.name}"
        target_path = TARGET_DIR / target_name

        shutil.copy2(image_path, target_path)
        copied_count += 1

    print(f"Kaynak klasör: {SOURCE_DIR}")
    print(f"OPEN_WHEEL sınıfına aktarılan görsel: {copied_count}")
    print(f"Bozuk olduğu için atlanan görsel: {skipped_broken}")
    print(f"Hedef klasör: {TARGET_DIR}")


if __name__ == "__main__":
    main()