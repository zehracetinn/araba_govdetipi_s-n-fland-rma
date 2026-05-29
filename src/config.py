from pathlib import Path

# Proje ana dizini
BASE_DIR = Path(__file__).resolve().parent.parent

# Ham veri kaynağı: arkadaştan alınan, 8 sınıfa göre düzenlenmiş klasör.
# Klasör isimleri küçük harf olduğu için CLASS_NAMES ile eşleştiriyoruz.
DATA_SOURCE_DIR = BASE_DIR / "data"

FOLDER_TO_CLASS = {
    "suv": "SUV",
    "van": "VAN",
    "station_wagon": "STATION_WAGON",
    "micro": "MICRO",
    "open_wheel": "OPEN_WHEEL",
    "sedan": "SEDAN",
    "hatchback": "HATCHBACK",
    "pickup": "PICK_UP",
}

# Dataset yolları
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Çıktı klasörü
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.pth"
CLASS_NAMES_PATH = OUTPUT_DIR / "class_names.json"

# Log klasörü (eğitim/değerlendirme/split adımlarının kayıtları)
LOGS_DIR = BASE_DIR / "logs"

# Veri bölme oranları (train/val/test = %70/%15/%15)
# Not: PDF'teki nihai test seti sunumda hocalar tarafından verilecektir.
# Buradaki test seti, kendi iç değerlendirmemiz (genelleme ölçümü) içindir.
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model ayarları
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
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