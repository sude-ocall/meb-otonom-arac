"""
komutlar.py
Beyin ↔ Motor arasında yayınlanan tüm string komutlarının tek kaynağı.

Hem ros2_beyin_yayin.py hem ros2_motor_dinleyici.py buradan import eder.
Magic string typo'ları artık import zamanında yakalanır:
  Komut.DDUR  → AttributeError (anında belli olur)
  "DDUR"      → çalışır, kimse durmaz, araç duvara çarpar
"""


class Komut:
    """ROS2 üzerinde gönderilen sabit komut string'leri."""
    DUR             = "DUR"
    YESIL_ISIK      = "YESIL_ISIK"
    BEKLEME_BITTI   = "BEKLEME_BITTI"
    SOLLAMA         = "SOLLAMA"
    SAGA_DON        = "SAGA_DON"
    PARK_TABELASI   = "PARK_TABELASI"
    PARK_ET         = "PARK_ET"
    KALP            = "KALP"   # Heartbeat — beyin yaşıyor sinyali

    @staticmethod
    def hiz(yuzde: int) -> str:
        """'HIZ:40' formatında hız komutu üretir."""
        return f"HIZ:{yuzde}"

    @staticmethod
    def hiz_parse(mesaj: str) -> int | None:
        """'HIZ:40' → 40, geçersizse None."""
        if not mesaj.startswith("HIZ:"):
            return None
        try:
            return int(mesaj.split(":", 1)[1])
        except (ValueError, IndexError):
            return None


class Topic:
    """ROS2 topic adları."""
    ARAC_KOMUT  = "/arac_komut"
    SERIT_SAPMA = "/serit_sapma"
    KALP_ATIS   = "/beyin_kalp"   # Watchdog için


# ── Hız profilleri ─────────────────────────────────────────────────────────
HIZ_NORMAL  = 40   # Düz parkur
HIZ_YAVAS   = 20   # Hız tümseği
HIZ_PARK    = 25   # Park alanı arama
HIZ_DONUS   = 30   # Çıkmaz yol dönüşü
HIZ_SOLLAMA = 35   # Sollama manevrası

# ── Watchdog ayarları ─────────────────────────────────────────────────────
KALP_HZ          = 5.0   # Beyin saniyede 5 kez heartbeat yayınlar
KALP_TIMEOUT_SN  = 1.0   # Bu sürede heartbeat gelmezse motor acil fren
