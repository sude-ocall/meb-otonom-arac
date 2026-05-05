"""
serit_beyni.py
Şerit takibi — kuş bakışı warp + sliding-window line tracking.

Kılavuz 3.4.8 (Bölge Tamamlama):
  Otonom araç şerit ihlali yapmadan ve parkurdan çıkmadan ilerlemeli.
  Tek histogramlı tespit virajlarda zayıf kalır → sliding-window kullanılır.

Akış:
  1. Perspektif warp (kuş bakışı)
  2. HLS L kanalında beyaz şerit maskesi
  3. Görüntüyü N yatay banda böl, her bantta sol/sağ peak ara
  4. Bant orta noktalarının ortalaması → istenen merkez
  5. Sapma = frame merkezi - şerit merkezi  (negatif: sola, pozitif: sağa)

Geriye uyumluluk: otonom_beyin(frame) → (frame, sapma:int) imzası korundu.
"""
import cv2
import numpy as np

# ── Sliding window parametreleri ──────────────────────────────────────────
# 9 bant → ~50 px yükseklik (480/9). Daha çok bant: pürüzsüz; daha az: gürültüye dirençli
BANT_SAYISI       = 9
PEAK_MIN_PIKSEL   = 30        # Bir bantta peak kabul için min piksel sayısı
ARAMA_PENCERE_GEN = 80        # Önceki banttan +/- bu kadar piksel arama yarıçapı

# HLS L kanalı eşiği — beyaz şeritler için
L_ESIK_ALT = 170
L_ESIK_UST = 255


def _maske_uret(warp):
    """HLS L kanalından beyaz şerit maskesi."""
    hls = cv2.cvtColor(warp, cv2.COLOR_BGR2HLS)
    return cv2.inRange(
        hls,
        np.array([0, L_ESIK_ALT, 0]),
        np.array([255, L_ESIK_UST, 255]),
    )


def _bant_peak_bul(bant_mask, son_x, w):
    """
    Bir banttaki şerit x-koordinatını döner.
    son_x: önceki bantta bulunan x (None ise tüm bantta ara).
    Hiç peak yoksa None döner.
    """
    if son_x is None:
        kolonlar = np.sum(bant_mask, axis=0)
    else:
        sol_sinir = max(0, son_x - ARAMA_PENCERE_GEN)
        sag_sinir = min(w, son_x + ARAMA_PENCERE_GEN)
        kolonlar = np.zeros(w, dtype=np.int32)
        kolonlar[sol_sinir:sag_sinir] = np.sum(
            bant_mask[:, sol_sinir:sag_sinir], axis=0
        )

    if kolonlar.max() < PEAK_MIN_PIKSEL:
        return None
    return int(np.argmax(kolonlar))


def otonom_beyin(frame):
    """
    Frame → (orijinal_frame, sapma).
      sapma > 0 → sol tarafa kaymış, sola düzeltme gerek (negatif yön komut)
      sapma < 0 → sağ tarafa kaymış
      sapma  ≈ 0 → ortada
    Geriye uyumluluk: imza/tip eski API ile aynı.
    """
    if frame is None:
        return None, 0

    h, w = frame.shape[:2]

    # ── 1. Perspektif warp (kuş bakışı) ────────────────────────────────────
    src = np.float32([
        [w * 0.40, h * 0.65],
        [w * 0.60, h * 0.65],
        [w * 0.05, h],
        [w * 0.95, h],
    ])
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(frame, M, (w, h))

    mask = _maske_uret(warp)

    # ── 2. Sliding window: alt banttan üst banta ──────────────────────────
    bant_h = h // BANT_SAYISI
    sol_x_son  = None
    sag_x_son  = None
    sol_xs, sag_xs = [], []
    mid = w // 2

    # Alttan başla (araca en yakın bant) — orada başlangıç peak'ini bulmak kolay
    for i in range(BANT_SAYISI - 1, -1, -1):
        y1 = i * bant_h
        y2 = (i + 1) * bant_h
        bant = mask[y1:y2, :]

        # SOL tarafı sadece sol yarıda ara (ilk bant), sonra önceki x etrafında
        sol_baslangic = sol_x_son if sol_x_son is not None else None
        if sol_baslangic is None:
            sol_kolonlar = np.sum(bant[:, :mid], axis=0)
            if sol_kolonlar.max() >= PEAK_MIN_PIKSEL:
                sol_baslangic = int(np.argmax(sol_kolonlar))

        if sol_baslangic is not None:
            sol_x = _bant_peak_bul(bant, sol_baslangic, w)
            if sol_x is not None and sol_x < mid + ARAMA_PENCERE_GEN:
                sol_xs.append(sol_x)
                sol_x_son = sol_x

        # SAĞ tarafı sadece sağ yarıda ara (ilk bant)
        sag_baslangic = sag_x_son if sag_x_son is not None else None
        if sag_baslangic is None:
            sag_kolonlar = np.zeros(w, dtype=np.int32)
            sag_kolonlar[mid:] = np.sum(bant[:, mid:], axis=0)
            if sag_kolonlar.max() >= PEAK_MIN_PIKSEL:
                sag_baslangic = int(np.argmax(sag_kolonlar))

        if sag_baslangic is not None:
            sag_x = _bant_peak_bul(bant, sag_baslangic, w)
            if sag_x is not None and sag_x > mid - ARAMA_PENCERE_GEN:
                sag_xs.append(sag_x)
                sag_x_son = sag_x

    # ── 3. Sapmayı hesapla ────────────────────────────────────────────────
    # Geriye uyumluluk: tek yan eksikse karşı sınırı frame kenarı kabul et
    # (orijinal histogram-tabanlı kodla aynı konvansiyon).
    if sol_xs and sag_xs:
        sol_ort = sum(sol_xs) / len(sol_xs)
        sag_ort = sum(sag_xs) / len(sag_xs)
        serit_merkez = (sol_ort + sag_ort) / 2
    elif sol_xs:
        # Sağ şerit görünmedi → sağ kenarı w kabul et
        sol_ort = sum(sol_xs) / len(sol_xs)
        serit_merkez = (sol_ort + w) / 2
    elif sag_xs:
        # Sol şerit görünmedi → sol kenarı 0 kabul et
        sag_ort = sum(sag_xs) / len(sag_xs)
        serit_merkez = sag_ort / 2
    else:
        # Hiç şerit görmedik — sapma 0, lane follower düz git der
        return frame, 0

    sapma = (w / 2) - serit_merkez
    return frame, int(sapma)
