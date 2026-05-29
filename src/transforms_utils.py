"""
Letterbox tabanlı preprocessing.

Neden letterbox?
Sedan↔StationWagon ve SUV↔Pickup çiftleri yalnızca aracın ARKA kısmında
ayrışır. Crop tabanlı yöntemler (CenterCrop, agresif RandomResizedCrop) bu
ayırt edici bölgeyi sık sık kesip attığı için model bu çiftleri karıştırıyordu.

Letterbox, aracın TAMAMINI en-boy oranını bozmadan (gri kenar dolgusuyla)
224x224 tuvale yerleştirir. Böylece arka kısım her zaman görünür kalır.
"""

import random

from PIL import Image, ImageOps


# ImageNet ortalamasına yakın nötr dolgu rengi (normalizasyon sonrası ~0).
PAD_COLOR = (124, 116, 104)


class CenterLetterbox:
    """
    Tahmin/değerlendirme için: görseli oranını koruyarak (size x size) tuvale
    sığdırır ve ortalar. Kırpma yapmaz -> tüm araç görünür.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.pad(
            img,
            (self.size, self.size),
            color=PAD_COLOR
        )


class RandomLetterbox:
    """
    Eğitim için: aracın tamamını koruyarak rastgele BOYUT (zoom) ve KONUMDA
    tuvale yerleştirir. Crop yapmadığı için aracın arka kısmı asla kesilmez;
    aynı zamanda çerçeve/oran ipucunu rastgeleleştirerek modelin 'geniş=sedan'
    gibi sahte kısayollar öğrenmesini engeller.
    """

    def __init__(self, size: int, min_scale: float = 0.6, max_scale: float = 1.0):
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        scale = random.uniform(self.min_scale, self.max_scale)
        target = max(1, int(self.size * scale))

        # Aracın tamamı hedef kareye sığacak şekilde küçültülür (oran korunur).
        fitted = ImageOps.contain(img, (target, target))

        canvas = Image.new("RGB", (self.size, self.size), PAD_COLOR)

        max_x = self.size - fitted.width
        max_y = self.size - fitted.height
        offset_x = random.randint(0, max(0, max_x))
        offset_y = random.randint(0, max(0, max_y))

        canvas.paste(fitted, (offset_x, offset_y))
        return canvas
