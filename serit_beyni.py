import cv2
import numpy as np

def otonom_beyin(frame):
    if frame is None: return None, 0
    h, w = frame.shape[:2]
    # Perspektif Dönüşümü (Kuş Bakışı) [cite: 581, 622]
    src = np.float32([[w*0.4, h*0.65], [w*0.6, h*0.65], [w*0.05, h], [w*0.95, h]])
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(frame, M, (w, h))
    # Şerit Maskeleme ve Histogram [cite: 583, 637]
    hls = cv2.cvtColor(warp, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, np.array([0, 170, 0]), np.array([255, 255, 255]))
    histogram = np.sum(mask[h//2:, :], axis=0)
    mid = len(histogram) // 2
    l_base = np.argmax(histogram[:mid]) if np.max(histogram[:mid]) > 0 else 0
    r_base = (np.argmax(histogram[mid:]) + mid) if np.max(histogram[mid:]) > 0 else w
    sapma = (w // 2) - ((l_base + r_base) // 2) [cite: 641]
    return frame, int(sapma)