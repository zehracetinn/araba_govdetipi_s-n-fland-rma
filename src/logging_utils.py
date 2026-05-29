"""
Basit log altyapısı.

Amaç: train / evaluate / split gibi scriptlerin tüm konsol çıktısını
hem ekrana yazmak hem de logs/ klasörü altına zaman damgalı bir dosyaya
kaydetmek. Böylece eğitim sürecini sonradan rapora aktarmak kolaylaşır.

Kullanım:
    from logging_utils import setup_logging
    setup_logging("train")   # main() başında bir kez çağrılır
"""

import sys
from datetime import datetime

from config import LOGS_DIR


class _Tee:
    """
    Yazılan veriyi birden fazla akışa (stream) aynı anda gönderir.
    Örn: hem terminale (sys.__stdout__) hem de log dosyasına.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(name: str):
    """
    stdout ve stderr'i bir log dosyasına da yönlendirir.

    Parametre:
        name: log dosyasının ön eki (örn: "train", "evaluate", "split")

    Döndürür:
        Oluşturulan log dosyasının yolu (Path).
    """

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{name}_{timestamp}.log"

    log_file = open(log_path, "w", encoding="utf-8")

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Log dosyası: {log_path}")
    print(f"Başlangıç zamanı: {timestamp}\n")

    return log_path
