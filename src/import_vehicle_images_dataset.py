import shutil
from pathlib import Path

from config import RAW_DIR


SOURCE_DIR = Path("downloads/vehicle_images")
TARGET_DIR = RAW_DIR / "MICRO"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_city_car_folder():
    for folder in SOURCE_DIR.rglob("*"):
        if folder.is_dir() and folder.name.lower() == "city car":
            return folder
    return None


def main():
    if not SOURCE_DIR.exists():
        print(f"Kaynak klasör bulunamadı: {SOURCE_DIR}")
        return

    city_car_dir = find_city_car_folder()

    if city_car_dir is None:
        print("City Car klasörü bulunamadı.")
        print("Kontrol için şunu çalıştır:")
        print("find downloads/vehicle_images -maxdepth 3 -type d")
        return

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        file for file in city_car_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]

    copied_count = 0

    for image_path in image_files:
        target_name = f"micro_citycar_{copied_count:05d}_{image_path.name}"
        target_path = TARGET_DIR / target_name

        if not target_path.exists():
            shutil.copy2(image_path, target_path)
            copied_count += 1

    print(f"Kaynak klasör: {city_car_dir}")
    print(f"MICRO sınıfına aktarılan görsel: {copied_count}")
    print(f"Hedef klasör: {TARGET_DIR}")


if __name__ == "__main__":
    main()