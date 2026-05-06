"""
serit_beyni.py
Şerit takibi — kuş bakışı warp + tek histogram peak (HIZLI sürüm).

Önceki sliding-window (9 bant) sürümü Pi 4'te kareyi 30-60 ms uzatıyordu;
USB kamera yayınında belirgin gecikme yaratıyordu. Bu yüzden Görev 8'de
küçük bir viraj sapma toleransı kaybı kabul edilip, eski tek-histogram
mantığına dönüldü (~5 ms).

Imza geriye uyumlu: otonom_beyin(frame) → (frame, sapma:int)
  sapma > 0 → şerit merkezi sola, sola düzeltme
  sapma < 0 → şerit merkezi sağa
  sapma ≈ 0 → ortada
"""
import cv2
import numpy as np

# HLS L kanalı eşiği — beyaz şeritler için
L_ESIK_ALT = 170
L_ESIK_UST = 255


def otonom_beyin(frame):
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

    # ── 2. Beyaz şerit maskesi (HLS L kanalı) ─────────────────────────────
    hls = cv2.cvtColor(warp, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(
        hls,
        np.array([0,   L_ESIK_ALT, 0]),
        np.array([255, L_ESIK_UST, 255]),
    )

    # ── 3. Tek histogram peak (en alt 1/8: araca en yakın bölge) ─────────
    # Daha büyük bir dilim almak warp distorsiyonu nedeniyle peak'i x=0/w'a
    # doğru kaydırır (üstte şerit dağılıp kenara yapışır). En alt ~60 satır
    # sliding-window'un tek bantına denk, peak doğru konumda çıkar.
    histogram = np.sum(mask[7 * h // 8:, :], axis=0)
    mid = len(histogram) // 2

    sol_var = histogram[:mid].max() > 0
    sag_var = histogram[mid:].max() > 0

    if sol_var and sag_var:
        l_base = int(np.argmax(histogram[:mid]))
        r_base = int(np.argmax(histogram[mid:])) + mid
        serit_merkez = (l_base + r_base) / 2
    elif sol_var:
        # Sağ şerit görünmüyor → sağ kenarı w kabul et, merkezi sola çek
        l_base = int(np.argmax(histogram[:mid]))
        serit_merkez = (l_base + w) / 2
    elif sag_var:
        # Sol şerit görünmüyor → sol kenarı 0 kabul et, merkezi sağa çek
        r_base = int(np.argmax(histogram[mid:])) + mid
        serit_merkez = r_base / 2
    else:
        return frame, 0

    sapma = (w / 2) - serit_merkez
    return frame, int(sapma)
