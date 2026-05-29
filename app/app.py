"""
Araba Gövde Tipi Sınıflandırma - Streamlit Web Arayüzü

Çalıştırma:
    ./venv/bin/streamlit run app/app.py

İçerik (PDF'teki arayüz isterleri):
    1) Görüntü yükleme alanı + önizleme
    2) "Sınıflandır" butonu
    3) Tahmin sonucu (büyük sınıf adı) + güven skoru
    4) Tüm sınıfların olasılık dağılımı (bar chart)
    5) Yüklenen görsel ile sonucun yan yana gösterimi
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# src/ klasörünü import yoluna ekliyoruz ki model ve config'i kullanabilelim.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from config import MODEL_PATH, IMAGE_SIZE, DISPLAY_NAMES  # noqa: E402
from model import create_model  # noqa: E402
from transforms_utils import CenterLetterbox  # noqa: E402


st.set_page_config(
    page_title="Araba Gövde Tipi Sınıflandırma",
    page_icon="🚗",
    layout="wide"
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transform():
    """
    Eğitimdeki validation/test preprocessing ile birebir aynı.
    Tahmin aşamasında augmentation YOK.
    """
    return transforms.Compose([
        CenterLetterbox(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


@st.cache_resource
def load_model():
    """
    Model bir kez yüklenir ve cache'lenir (her tahminde tekrar yüklenmez -> hız).
    """
    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    class_names = checkpoint["class_names"]

    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def predict(image, model, class_names, device, transform):
    """
    Tek bir PIL görseli için tahmin yapar.
    Döndürür: (tahmin_sınıfı, güven_skoru, tüm_olasılıklar_dict)
    """
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    probs = probabilities.cpu().numpy()

    predicted_index = int(probs.argmax())
    predicted_class = class_names[predicted_index]
    confidence = float(probs[predicted_index])

    prob_dict = {
        DISPLAY_NAMES.get(name, name): float(prob)
        for name, prob in zip(class_names, probs)
    }

    return predicted_class, confidence, prob_dict


# -------------------- Arayüz --------------------

st.title("🚗 Araba Gövde Tipi Sınıflandırma")
st.caption(
    "MobileNetV3-Large tabanlı model · 8 sınıf: "
    "SUV, VAN, STATION WAGON, MICRO, AÇIK TEKERLEKLİ, SEDAN, HATCHBACK, PICK UP"
)

# Model yüklenemiyorsa kullanıcıyı bilgilendir.
if not MODEL_PATH.exists():
    st.error(
        f"Eğitilmiş model bulunamadı: {MODEL_PATH}\n\n"
        "Önce eğitimi çalıştırın: `./venv/bin/python src/train.py`"
    )
    st.stop()

model, class_names, device = load_model()

# Bölüm 1: Görüntü yükleme alanı
uploaded_file = st.file_uploader(
    "Bir araba görseli yükleyin",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Bölüm 4: Yüklenen görsel ile sonuç yan yana
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Yüklenen Görüntü")
        st.image(image, use_container_width=True)

    # Bölüm 2: Tahmin butonu
    with right_column:
        st.subheader("Tahmin Sonucu")

        if st.button("🔍 Sınıflandır", type="primary", use_container_width=True):
            transform = get_transform()

            import time
            start = time.time()
            predicted_class, confidence, prob_dict = predict(
                image, model, class_names, device, transform
            )
            elapsed_ms = (time.time() - start) * 1000

            display_name = DISPLAY_NAMES.get(predicted_class, predicted_class)

            # Bölüm 3: Büyük ve belirgin sonuç + güven skoru
            st.markdown(f"## 🏆 {display_name}")
            st.metric("Güven Skoru", f"%{confidence * 100:.2f}")
            st.caption(f"Tahmin süresi: {elapsed_ms:.0f} ms")

            # Bölüm 3 (devam): Tüm sınıflar için olasılık dağılımı (bar chart)
            st.subheader("Tüm Sınıfların Olasılık Dağılımı")
            prob_df = pd.DataFrame(
                {"Olasılık": prob_dict}
            ).sort_values("Olasılık", ascending=False)
            st.bar_chart(prob_df)
        else:
            st.info("Tahmin için 'Sınıflandır' butonuna basın.")
else:
    st.info("Başlamak için yukarıdan bir araba görseli yükleyin.")
