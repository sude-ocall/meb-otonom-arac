"""
test_tespit.py
Canlı tespit tanı aracı.

Kameraya tabela, trafik ışığı veya telefondan görsel tut.
Her tespit kutu + sınıf adı + confidence olarak ekranda görünür.
Güven eşiği yok — modelin gördüğü HER şeyi gösterir.

Çalıştır:
    .venv/bin/python test_tespit.py

Tuşlar:
    q          → çıkış
    + / -      → eşiği 0.05 artır / azalt
"""
import cv2
import sys
from ultralytics import YOLO

MODEL_YOLU   = "best.pt"
KAMERA_INDEX = 0

# Renk paleti — sınıfa göre kutu rengi
RENK_TABLOSU = {
    "yesil"      : (0,   200,  0),
    "kirmizi"    : (0,     0, 220),
    "sari"       : (0,   200, 200),
    "YayaGecidi" : (0,   140, 255),
    "dur"        : (0,     0, 200),
}
VARSAYILAN_RENK = (200, 200, 200)

print("Model yükleniyor...")
try:
    model = YOLO(MODEL_YOLU)
except Exception as e:
    print(f"HATA: {e}")
    sys.exit(1)

print("Modeldeki sınıflar:", list(model.names.values()))
print("Kamera açılıyor...\n")

cap = cv2.VideoCapture(KAMERA_INDEX)
if not cap.isOpened():
    print(f"Kamera {KAMERA_INDEX} açılamadı.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

esik = 0.25   # Başlangıç eşiği — ekranda gösterilen minimum confidence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Model çalıştır (eşik yok — hepsini al) ────────────────────────────
    results = model(frame, verbose=False)[0]

    en_yuksek_conf = 0.0
    tespit_sayisi  = 0

    for box in results.boxes:
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        ad     = results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Eşiğin altındakiler gri, üstündekiler renkli göster
        if conf >= esik:
            renk = RENK_TABLOSU.get(ad, VARSAYILAN_RENK)
            kalinlik = 2
            tespit_sayisi += 1
            en_yuksek_conf = max(en_yuksek_conf, conf)
        else:
            renk = (80, 80, 80)   # Gri = eşik altı, gözlem için
            kalinlik = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, kalinlik)
        cv2.putText(
            frame, f"{ad}  {conf:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1
        )

    # ── Bilgi çubuğu ──────────────────────────────────────────────────────
    durum = (
        f"Esik: {esik:.2f}  |  "
        f"Aktif tespit: {tespit_sayisi}  |  "
        f"En yuksek conf: {en_yuksek_conf:.2f}  |  "
        f"+/- ile esigi degistir"
    )
    cv2.rectangle(frame, (0, 0), (640, 26), (0, 0, 0), -1)
    cv2.putText(frame, durum, (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    # ── Terminale de yaz (eşiği geçenler) ─────────────────────────────────
    if tespit_sayisi > 0:
        satir = "  |  ".join(
            f"{results.names[int(b.cls[0])]} {float(b.conf[0]):.2f}"
            for b in results.boxes
            if float(b.conf[0]) >= esik
        )
        print(f"[TESPİT] {satir}")

    cv2.imshow("Canli Tespit Tani", frame)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('q'):
        break
    elif tus == ord('+') and esik < 0.95:
        esik = round(esik + 0.05, 2)
        print(f"Esik → {esik}")
    elif tus == ord('-') and esik > 0.05:
        esik = round(esik - 0.05, 2)
        print(f"Esik → {esik}")

cap.release()
cv2.destroyAllWindows()
