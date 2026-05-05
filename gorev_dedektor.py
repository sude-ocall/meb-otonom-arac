"""
gorev_dedektor.py
YOLOv8 model'inde sınıfı bulunmayan görevleri HSV/şekil tabanlı tespit eder.

Kılavuz 3.4 referansı:
  - Görev 3 (Hız Tümseği)   → Şekil 5: sarı-siyah dikey şeritler, 1000×300 mm
  - Görev 4 (Hemzemin Geçit)→ Şekil 4: beyaz ızgara deseni, 1000×400 mm
  - Görev 5 (Sollama)       → Turuncu araç (20×30×25 cm) önümüzde

Bu modül nesne_beyni.tahmin_yap() ile PARALEL çalışır; ros2_beyin_yayin
NORMAL durumunda her ikisini birden sorgular.
"""
import cv2
import numpy as np


# ── HSV renk aralıkları ────────────────────────────────────────────────────
# OpenCV: H 0-179, S 0-255, V 0-255

# Sarı (hız tümseği şeritleri)
SARI_ALT = np.array([22, 100, 100])
SARI_UST = np.array([35, 255, 255])

# Turuncu (sollanacak araç) — sarıdan ayırt etmek için H sınırı dar
TURUNCU_ALT = np.array([8,  150, 120])
TURUNCU_UST = np.array([20, 255, 255])

# Beyaz (hemzemin geçit blokları, şerit çizgileri)
BEYAZ_ALT = np.array([0,    0, 200])
BEYAZ_UST = np.array([180, 60, 255])

# Mavi (Görev 8 bölge tamamlama işaretleri — ileride lazım olursa)
MAVI_ALT = np.array([100, 120, 80])
MAVI_UST = np.array([130, 255, 255])


# ──────────────────────────────────────────────────────────────────────────
# Görev 3: Hız Tümseği (Şekil 5)
# ──────────────────────────────────────────────────────────────────────────
def hiz_tumsek_var_mi(frame, min_serit: int = 4) -> tuple[bool, float]:
    """
    Sarı-siyah dikey şerit deseni var mı?

    Şekil 5 → 60mm sarı + 57.5mm siyah, dikey, 300mm yükseklik.
    Min_serit kadar sarı dikey şerit yan yana → hız tümseği.

    Döner: (var_mi, yakinlik)
        yakinlik: 0.0 = çok uzak, 1.0 = aracın hemen önünde
                  (tümseğin en alt y / frame yüksekliği)
    """
    h, w = frame.shape[:2]
    bolge = frame[h // 2:, :]                       # zemin tarafı
    hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, SARI_ALT, SARI_UST)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    serit_sayisi = 0
    en_alt_y = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        alan = cv2.contourArea(c)
        # Dikey şerit kriteri: yükseklik > genişlik, anlamlı boyut
        if ch > cw and ch > 18 and alan > 250:
            serit_sayisi += 1
            en_alt_y = max(en_alt_y, y + ch + h // 2)

    if serit_sayisi < min_serit:
        return False, 0.0

    yakinlik = en_alt_y / h
    return True, yakinlik


# ──────────────────────────────────────────────────────────────────────────
# Görev 4: Hemzemin Geçit (Şekil 4)
# ──────────────────────────────────────────────────────────────────────────
def hemzemin_var_mi(frame, min_blok: int = 8) -> tuple[bool, float]:
    """
    Beyaz ızgara (kare blok) deseni var mı?

    Şekil 4 → 60mm × 100mm beyaz bloklar, 40mm gap, 2 sıra.
    En az min_blok kare blok bulunursa hemzemin sayılır.

    Döner: (var_mi, yakinlik)
    """
    h, w = frame.shape[:2]
    bolge = frame[h // 2:, :]
    hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, BEYAZ_ALT, BEYAZ_UST)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blok_sayisi = 0
    en_alt_y = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        alan = cv2.contourArea(c)
        # Kare/dikdörtgen blok kriteri (60×100 mm projeksiyon ~0.4-1.5)
        if ch == 0:
            continue
        oran = cw / ch
        if 80 < alan < 4000 and 0.35 < oran < 1.6:
            blok_sayisi += 1
            en_alt_y = max(en_alt_y, y + ch + h // 2)

    if blok_sayisi < min_blok:
        return False, 0.0

    yakinlik = en_alt_y / h
    return True, yakinlik


# ──────────────────────────────────────────────────────────────────────────
# Görev 5: Sollama — Turuncu Araç Tespiti
# ──────────────────────────────────────────────────────────────────────────
# Kılavuz 3.1: "Sollanacak araç ... yalnızca sollama yasağının olmadığı
# bölgelerden herhangi birine yerleştirilecektir." Yani turuncu görmek =
# sollama serbest bölgede olduğumuzun garantisi. Ayrıca tabela aramaya
# gerek yok.
# ──────────────────────────────────────────────────────────────────────────
def turuncu_arac_var_mi(frame,
                         min_alan_orani: float = 0.04
                         ) -> tuple[bool, int, int]:
    """
    Önümüzdeki turuncu aracı tespit eder.

    Sadece frame'in orta-alt 2/3'üne bakar (gökyüzündeki turuncuları yok say).
    min_alan_orani: bölge alanına oranla minimum turuncu blok büyüklüğü
                    (0.04 → bölgenin en az %4'ü turuncu olmalı)

    Döner: (var_mi, alan_pikseli, x_merkez_tam_frame)
    """
    h, w = frame.shape[:2]
    # İlgi bölgesi: üst %25 (gökyüzü) yok say, sağ-sol kenar yok say
    y_bas = int(h * 0.25)
    x_bas, x_son = int(w * 0.15), int(w * 0.85)
    bolge = frame[y_bas:, x_bas:x_son]
    hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, TURUNCU_ALT, TURUNCU_UST)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False, 0, 0

    en_buyuk = max(contours, key=cv2.contourArea)
    alan = int(cv2.contourArea(en_buyuk))

    bolge_alani = bolge.shape[0] * bolge.shape[1]
    if alan < int(bolge_alani * min_alan_orani):
        return False, alan, 0

    M = cv2.moments(en_buyuk)
    if M["m00"] == 0:
        return False, alan, 0

    cx_lokal = int(M["m10"] / M["m00"])
    cx_tam = cx_lokal + x_bas
    return True, alan, cx_tam


# ──────────────────────────────────────────────────────────────────────────
# Yardımcı: ileride Görev 8 için (mavi bölge işareti)
# ──────────────────────────────────────────────────────────────────────────
def mavi_bolge_isareti_var_mi(frame,
                                min_alan: int = 800
                                ) -> bool:
    """
    Yol kenarına yerleştirilen mavi bölge işaretlerini tespit eder.
    Görev 8 sayacını ilerletmek için kullanılabilir (henüz aktif değil).
    """
    h, w = frame.shape[:2]
    # Üst yarı + kenarlar (yol kenarındaki dikey işaretler)
    sol = frame[: 2 * h // 3, : w // 4]
    sag = frame[: 2 * h // 3, 3 * w // 4 :]

    for bolge in (sol, sag):
        hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, MAVI_ALT, MAVI_UST)
        if cv2.countNonZero(mask) >= min_alan:
            return True
    return False
