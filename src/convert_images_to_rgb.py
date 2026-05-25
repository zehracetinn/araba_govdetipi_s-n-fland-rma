from pathlib import Path
from PIL import Image

from config import RAW_DIR


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def convert_image_to_rgb(image_path: Path):
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")

            # Dosya uzantısı ne olursa olsun güvenli şekilde JPEG olarak kaydediyoruz.
            # Eski dosya aynı isimle kalmasın diye .jpg yapıyoruz.
            new_path = image_path.with_suffix(".jpg")

            rgb_img.save(new_path, "JPEG", quality=95)

        # Eğer eski dosya jpg değilse ve yeni dosya oluştuysa eskiyi silebiliriz.
        if image_path != new_path and new_path.exists():
            image_path.unlink()

        return True

    except Exception as e:
        print(f"Hata: {image_path} -> {e}")
        return False


def main():
    converted_count = 0
    error_count = 0

    image_files = [
        file for file in RAW_DIR.rglob("*")
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"Toplam görsel bulundu: {len(image_files)}")

    for image_path in image_files:
        success = convert_image_to_rgb(image_path)

        if success:
            converted_count += 1
        else:
            error_count += 1

    print("\nRGB dönüşüm tamamlandı.")
    print(f"Dönüştürülen görsel: {converted_count}")
    print(f"Hatalı görsel: {error_count}")


if __name__ == "__main__":
    main()