from config import RAW_DIR, CLASS_NAMES

def main():
    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Oluşturuldu: {class_dir}")

    print("\nRaw dataset klasörleri hazır.")

if __name__ == "__main__":
    main()