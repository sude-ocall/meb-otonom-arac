import cv2
import numpy as np

# ─── Renk maskeleri ──────────────────────────────────────────

def _kirmizi_maske(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,  100, 70]), np.array([10,  255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255]))
    return cv2.bitwise_or(m1, m2)

def _mavi_maske(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([100, 100, 50]), np.array([130, 255, 255]))

def _sari_maske(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

# ─── Şekil analizi ───────────────────────────────────────────

def _sekil_tani(cnt):
    """Konturun geometrik şeklini döner."""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    n = len(approx)
    if n == 3:
        return 'UCGEN'
    if n == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        oran = float(w) / h if h > 0 else 0
        return 'KARE' if 0.8 < oran < 1.2 else 'DIKDORTGEN'
    alan = cv2.contourArea(cnt)
    if peri > 0 and (4 * np.pi * alan / (peri ** 2)) > 0.65:
        return 'DAIRE'
    return 'DIGER'

def _en_buyuk_kontur(mask, min_alan):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    return cnt if cv2.contourArea(cnt) >= min_alan else None

# ─── Kırmızı üçgen tabelalar ─────────────────────────────────
# Yaya geçidi, hemzemin geçit ve hız tümseği hepsi kırmızı üçgen
# içerik analizi ile ayırt edilir:
#   Hemzemin: ızgara → çok sayıda kuvvetli yatay kenar çizgisi
#   Hız tümseği: sarı-siyah bantlı zemin → yüksek sarı piksel oranı
#   Yaya geçidi: varsayılan (dikey koyu figür, beyaz arka plan)

def _ic_kirmizi_analiz(ic):
    if ic.size == 0:
        return 'YAYA'

    # Hız tümseği: sarı bantlı (Şekil 5 - sarı-siyah çizgiler)
    sari = _sari_maske(ic)
    sari_oran = cv2.countNonZero(sari) / float(ic.shape[0] * ic.shape[1])
    if sari_oran > 0.12:
        return 'HIZ_TUMSEĞI'

    # Hemzemin geçit: yoğun yatay çizgi deseni (Şekil 4 - ızgara)
    gray = cv2.cvtColor(ic, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h = edges.shape[0]
    if h > 0:
        yogunluklar = [int(np.sum(edges[i, :])) for i in range(h)]
        maks = max(yogunluklar) if yogunluklar else 0
        esik = maks * 0.35
        kuvvetli = sum(1 for v in yogunluklar if v > esik)
        if kuvvetli >= 4:
            return 'HEMZEMIN'

    return 'YAYA'

def kirmizi_ucgen_tani(frame, min_alan=450):
    """
    Kırmızı çerçeveli üçgen tabelayı tanır.
    Döner: 'YAYA' | 'HEMZEMIN' | 'HIZ_TUMSEĞI' | None
    """
    mask = _kirmizi_maske(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnt = _en_buyuk_kontur(mask, min_alan)
    if cnt is None or _sekil_tani(cnt) != 'UCGEN':
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    ic = frame[y:y+h, x:x+w]
    return _ic_kirmizi_analiz(ic)

# ─── Mavi dikdörtgen tabelalar ───────────────────────────────
# Çıkmaz yol: mavi zemin, beyaz "T" ← üstte geniş yatay bar, altta az beyaz
# Park: mavi zemin, beyaz "P"  ← üst ve alt daha dengeli

def _ic_mavi_analiz(ic):
    gray = cv2.cvtColor(ic, cv2.COLOR_BGR2GRAY)
    _, beyaz = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    h, w = beyaz.shape
    if h == 0 or w == 0:
        return 'PARK'
    ust = float(np.sum(beyaz[:h // 3, :]))
    alt = float(np.sum(beyaz[2 * h // 3:, :]))
    # T: üstte ağır, altta çok az beyaz
    if ust > 0 and (alt / (ust + 1)) < 0.35:
        return 'CIKMAZ'
    return 'PARK'

def mavi_tabela_tani(frame, min_alan=300):
    """
    Mavi dikdörtgen tabelayı tanır.
    Döner: 'PARK' | 'CIKMAZ' | None
    """
    mask = _mavi_maske(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnt = _en_buyuk_kontur(mask, min_alan)
    if cnt is None:
        return None
    if _sekil_tani(cnt) not in ('DIKDORTGEN', 'KARE'):
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    ic = frame[y:y+h, x:x+w]
    return _ic_mavi_analiz(ic)

# ─── Sollama serbest tabelası ─────────────────────────────────
# Dairevi mavi tabela (Şekil 9 - iki araç içerir)

def sollama_serbest_var_mi(frame, min_alan=300):
    """Dairevi mavi tabela → sollama serbest bölgesi başlangıcı."""
    mask = _mavi_maske(frame)
    cnt = _en_buyuk_kontur(mask, min_alan)
    if cnt is not None and _sekil_tani(cnt) == 'DAIRE':
        return True
    return False

# ─── Ana arayüz ──────────────────────────────────────────────

def tabela_tani(frame):
    """
    Karede görünen tabelayı tek çağrıyla tanır.
    Döner: 'YAYA' | 'HEMZEMIN' | 'HIZ_TUMSEĞI' |
           'CIKMAZ' | 'PARK' | 'SOLLAMA' | None
    """
    sonuc = kirmizi_ucgen_tani(frame)
    if sonuc:
        return sonuc

    sonuc = mavi_tabela_tani(frame)
    if sonuc:
        return sonuc

    if sollama_serbest_var_mi(frame):
        return 'SOLLAMA'

    return None
