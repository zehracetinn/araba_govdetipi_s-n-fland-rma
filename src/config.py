from pathlib import Path

# Proje ana dizini
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset yolları
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# Çıktı klasörü
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.pth"
CLASS_NAMES_PATH = OUTPUT_DIR / "class_names.json"

# Model ayarları
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
NUM_CLASSES = 8

# Sınıf isimleri
CLASS_NAMES = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "OPEN_WHEEL",
    "SEDAN",
    "HATCHBACK",
    "PICK_UP"
]

# Kullanıcıya gösterilecek daha okunabilir isimler
DISPLAY_NAMES = {
    "SUV": "SUV",
    "VAN": "VAN",
    "STATION_WAGON": "STATION WAGON",
    "MICRO": "MICRO",
    "OPEN_WHEEL": "AÇIK TEKERLEKLİ",
    "SEDAN": "SEDAN",
    "HATCHBACK": "HATCHBACK",
    "PICK_UP": "PICK UP"
}