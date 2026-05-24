import shutil
from pathlib import Path

from config import RAW_DIR


SOURCE_DIRS = [
    Path("stanford_archive"),
    Path("Cars_Body_Type")
]

TARGET_DIR = RAW_DIR / "MICRO"

MICRO_KEYWORDS = [
    "smart fortwo",
    "smart-fortwo",
    "fortwo",
    "twizy",
    "isetta",
    "toyota-iq",
    "toyota iq"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_micro_image(file_path: Path) -> bool:
    name = file_path.name.lower()
    return any(keyword in name for keyword in MICRO_KEYWORDS)


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            print(f"Kaynak klasör bulunamadı: {source_dir}")
            continue

        image_files = [
            file for file in source_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        ]

        for image_path in image_files:
            if is_micro_image(image_path):
                target_name = f"micro_{copied_count:05d}_{image_path.name}"
                target_path = TARGET_DIR / target_name

                if not target_path.exists():
                    shutil.copy2(image_path, target_path)
                    copied_count += 1
                else:
                    skipped_count += 1

    print(f"MICRO sınıfına aktarılan görsel: {copied_count}")
    print(f"Atlanan görsel: {skipped_count}")
    print(f"Hedef klasör: {TARGET_DIR}")


if __name__ == "__main__":
    main()