"""
buton.py
Görev 1 — Yarışma başlatma butonu (kılavuz 3.4.1, +50 ödül puanı).

Kılavuz:
  "Otonom aracı, yarışma öncesinde yüklenmiş yazılım üzerinden, harici bir
   bilgisayar bağlantısına ihtiyaç duymadan, buton veya benzeri bir tetikleme
   elemanı ile çalıştırmaya hazır hâle getiren takımlara 50 ödül puanı verilir."

Donanım:
  Push-button → Pi GPIO 26 ile GND arası bağlanır.
  Pi'nin dahili pull-up direnci kullanıldığı için ek direnç gerekmez.
  Buton basılı değil → GPIO HIGH;   basılı → GPIO LOW

Kullanım:
  from buton import buton_basildi_mi_bekle, buton_kullanilabilir_mi
  if buton_kullanilabilir_mi():
      buton_basildi_mi_bekle()   # Buton basılana kadar blokla
"""
import time

BUTON_PIN          = 26      # Pi GPIO BCM numarası
DEBOUNCE_SN        = 0.05    # Bounce filtreleme (50 ms)
GERI_DONUS_OK_SN   = 30      # Bu süre içinde basılmazsa fallback (geliştirme)

_GPIO = None        # Lazy import — yalnızca Pi'da yüklenir
_setup_yapildi      = False


def _gpio_yukle():
    """RPi.GPIO modülünü tek seferlik yükle. Yüklenemezse None döner."""
    global _GPIO
    if _GPIO is not None:
        return _GPIO
    try:
        import RPi.GPIO as GPIO     # type: ignore
        _GPIO = GPIO
        return _GPIO
    except (ImportError, RuntimeError):
        return None


def buton_kullanilabilir_mi() -> bool:
    """RPi.GPIO yüklenebiliyor mu? (Mac/PC üzerinde False döner)"""
    return _gpio_yukle() is not None


def buton_hazirla() -> bool:
    """
    GPIO pinini giriş + dahili pull-up olarak ayarlar.
    Başarılı → True, donanım yoksa → False (sessizce skip).
    """
    global _setup_yapildi
    GPIO = _gpio_yukle()
    if GPIO is None:
        return False
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(BUTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        _setup_yapildi = True
        return True
    except Exception as e:
        print(f"[BUTON] GPIO setup hatası: {e}")
        return False


def buton_basili_mi() -> bool:
    """Anlık buton durumu — basılı (LOW) ise True."""
    GPIO = _gpio_yukle()
    if GPIO is None or not _setup_yapildi:
        return False
    return GPIO.input(BUTON_PIN) == GPIO.LOW


def buton_basildi_mi_bekle(zaman_asimi_sn: float | None = None) -> bool:
    """
    Buton basılana kadar bloklayarak bekle (debounce ile).

    zaman_asimi_sn:
      None   → süresiz bekle
      sayı   → bu kadar saniye sonra False ile çık (testlerde kullanışlı)

    Döner:
      True   → buton basıldı
      False  → zaman aşımı veya GPIO hazır değil
    """
    if not buton_kullanilabilir_mi():
        print("[BUTON] RPi.GPIO yok — buton beklemeden devam ediliyor (test modu).")
        return False

    if not _setup_yapildi and not buton_hazirla():
        print("[BUTON] Buton hazırlanamadı — devam ediliyor.")
        return False

    print(f"[BUTON] Yarışmayı başlatmak için butona bas (GPIO {BUTON_PIN})...")

    baslangic = time.time()
    while True:
        if buton_basili_mi():
            time.sleep(DEBOUNCE_SN)
            if buton_basili_mi():
                print("[BUTON] Basıldı — yarışma sürecine geçiliyor.")
                return True

        if zaman_asimi_sn is not None and (time.time() - baslangic) >= zaman_asimi_sn:
            print(f"[BUTON] {zaman_asimi_sn} sn zaman aşımı — devam.")
            return False

        time.sleep(0.02)


def buton_temizle() -> None:
    """Çıkışta GPIO temizliği — yalnızca biz başlattıysak."""
    global _setup_yapildi
    GPIO = _gpio_yukle()
    if GPIO is not None and _setup_yapildi:
        try:
            GPIO.cleanup(BUTON_PIN)
        except Exception:
            pass
        _setup_yapildi = False
