import shutil
from pathlib import Path

import pandas as pd

from config import BASE_DIR, RAW_DIR


CSV_PATH = BASE_DIR / "stanford_archive" / "stanford_cars_type.csv"
IMAGE_DIR = BASE_DIR / "stanford_archive" / "stanford_cars_type"

TARGET_CLASS = "STATION_WAGON"
TARGET_CAR_TYPE = "Wagon"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def build_image_index():
    """
    Stanford görsel klasöründeki tüm resimleri isimlerine göre indeksler.
    Böylece dosya alt klasörde olsa bile bulabiliriz.
    """
    image_index = {}

    if not IMAGE_DIR.exists():
        print(f"Görsel klasörü bulunamadı: {IMAGE_DIR}")
        return image_index

    for image_path in IMAGE_DIR.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            image_index[image_path.name] = image_path

    return image_index


def main():
    if not CSV_PATH.exists():
        print(f"CSV bulunamadı: {CSV_PATH}")
        return

    image_index = build_image_index()

    if not image_index:
        print("Hiç görsel bulunamadı.")
        return

    df = pd.read_csv(CSV_PATH)

    wagon_df = df[df["car_type"] == TARGET_CAR_TYPE]

    target_dir = RAW_DIR / TARGET_CLASS
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_count = 0

    for _, row in wagon_df.iterrows():
        possible_names = [
            str(row["file_name"]),
            str(row["new_filename"]),
        ]

        source_path = None

        for name in possible_names:
            if name in image_index:
                source_path = image_index[name]
                break

        if source_path is None:
            missing_count += 1
            continue

        new_file_name = f"stanford_wagon_{copied_count + 1:05d}{source_path.suffix.lower()}"
        target_path = target_dir / new_file_name

        shutil.copy2(source_path, target_path)
        copied_count += 1

    print(f"Stanford Wagon -> {TARGET_CLASS}")
    print(f"Aktarılan görsel: {copied_count}")
    print(f"Bulunamayan görsel: {missing_count}")


if __name__ == "__main__":
    main()