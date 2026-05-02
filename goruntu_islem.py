"""
goruntu_islem.py
Kamera görüntüsü ön işleme: parlama azaltma, kontrast eşitleme, renk izolasyonu.

Kullanım:
    from goruntu_islem import on_isle, roi_kirp, kirmizi_var_mi, mavi_var_mi
"""
import cv2
import numpy as np

# ── CLAHE — tek seferlik oluştur, her frame'de yeniden yaratma ─────────────
# clipLimit: kaç kat kontrast artışına izin verilsin (2.0 standart, parlama çoksa 3.0)
# tileGridSize: görüntüyü kaç parçaya bölerek işle (8×8 Jetson için dengeli)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ── 1. Parlama azaltma + kontrast eşitleme ─────────────────────────────────

def parlama_azalt(frame):
    """
    Güneş parlaması ve aşırı pozlamayı azaltır.

    Neden LAB?
      LAB'de L kanalı sadece parlaklığı tutar; renk bilgisi (a, b) korunur.
      L'ye CLAHE uygulanınca kontrast düzelir ama renkler bozulmaz.
      HSV veya BGR'ye uygulanırsa renkler kayar.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_esitlenmis = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_esitlenmis, a, b]), cv2.COLOR_LAB2BGR)


def on_isle(frame, clahe: bool = True):
    """
    Tam ön işleme zinciri.

    clahe=True  → parlama azaltma aktif (daha sağlam, ~2ms ekstra)
    clahe=False → ham frame döner (Jetson FPS kritikse False kullan)
    """
    if clahe:
        frame = parlama_azalt(frame)
    return frame


# ── 2. ROI — tabelaların bulunduğu dikey bant ──────────────────────────────

def roi_kirp(frame, ust_oran: float = 0.10, alt_oran: float = 0.78):
    """
    Tabelaların bulunduğu bölgeyi kırparak döndürür.

    Varsayılan: yüksekliğin %10-78'i arası
      - %0-10  → gökyüzü / tavan (tabela yok, gürültü var)
      - %78-100 → zemin / şerit (tabela yok; KirmiziPark ayrı işlenir)

    Döner: (kirpilmis_frame, y_ofseti)
    y_ofseti YOLO bbox koordinatlarını orijinal frame'e geri çevirmek için kullanılır.

    Örnek entegrasyon (nesne_beyni.tahmin_yap içinde):
        roi, y_ofset = roi_kirp(frame)
        results = model(roi)[0]
        # bbox'ları orijinal koordinata çevir:
        y1 += y_ofset; y2 += y_ofset
    """
    h = frame.shape[0]
    y1 = int(h * ust_oran)
    y2 = int(h * alt_oran)
    return frame[y1:y2, :].copy(), y1


# ── 3. Renk izolasyon maskeleri ────────────────────────────────────────────
# Bu fonksiyonlar doğrudan YOLO'ya geçirilmez.
# Amaç: YOLO tespitini HSV ile çapraz doğrulamak.

def kirmizi_var_mi(frame, x1, y1, x2, y2, min_oran: float = 0.10) -> bool:
    """
    Bbox bölgesinde anlamlı kırmızı piksel var mı?

    Kullanım: YOLO "dur" / "YayaGecidi" dediğinde false-positive öldürmek için.
    min_oran: bbox alanının en az kaç %'i kırmızı olmalı (varsayılan %10).
    """
    bolge = frame[max(0, y1):y2, max(0, x1):x2]
    if bolge.size == 0:
        return False
    hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, np.array([0,   100, 60]), np.array([10,  255, 255])) |
        cv2.inRange(hsv, np.array([165, 100, 60]), np.array([180, 255, 255]))
    )
    oran = cv2.countNonZero(mask) / (bolge.shape[0] * bolge.shape[1])
    return oran >= min_oran


def mavi_var_mi(frame, x1, y1, x2, y2, min_oran: float = 0.10) -> bool:
    """
    Bbox bölgesinde anlamlı mavi piksel var mı?

    Kullanım: Mavi tabela sınıflarını (SollamaSerbest vb.) doğrulamak için.
    """
    bolge = frame[max(0, y1):y2, max(0, x1):x2]
    if bolge.size == 0:
        return False
    hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 100, 60]), np.array([130, 255, 255]))
    oran = cv2.countNonZero(mask) / (bolge.shape[0] * bolge.shape[1])
    return oran >= min_oran


def sari_var_mi(frame, x1, y1, x2, y2, min_oran: float = 0.08) -> bool:
    """
    Bbox bölgesinde anlamlı sarı piksel var mı?

    Kullanım: Sarı tabelaları (HizTumseği vb.) doğrulamak için.
    """
    bolge = frame[max(0, y1):y2, max(0, x1):x2]
    if bolge.size == 0:
        return False
    hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 100, 80]), np.array([38, 255, 255]))
    oran = cv2.countNonZero(mask) / (bolge.shape[0] * bolge.shape[1])
    return oran >= min_oran
