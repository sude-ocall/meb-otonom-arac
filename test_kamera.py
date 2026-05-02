"""
test_kamera.py
MacBook web kamerasıyla YOLOv8 tabela/ışık tespitini simüle eder.
Motor sinyali gönderilmez; tüm kararlar terminale yazdırılır.

Çalıştır:
    .venv/bin/python test_kamera.py
Çıkış:
    q tuşuna bas veya pencereyi kapat.
"""
import cv2
import time
import sys
import os

# Proje kökünü path'e ekle (başka dizinden çalıştırılırsa da import çalışsın)
sys.path.insert(0, os.path.dirname(__file__))

from nesne_beyni import modeli_yukle, tahmin_yap, gorsele_ciz, DURMA_SINIFLARI

# ── Ayarlar ────────────────────────────────────────────────────────────────
MODEL_YOLU      = "best.pt"
KAMERA_INDEX    = 0         # MacBook dahili kamera
FRAME_GENISLIK  = 640
FRAME_YUKSEKLIK = 480
TABELA_COOLDOWN = 10        # Aynı tabelaya tepki vermeden beklenen süre (sn)

# Ekranda gösterilecek renk paleti
RENK = {
    "default": (0, 255, 0),     # Yeşil  — normal tespit
    "uyari"  : (0, 165, 255),   # Turuncu — dur/yaya/hemzemin
    "bilgi"  : (255, 255, 0),   # Sarı   — bilgi metni
}

# ── Model yükleme ──────────────────────────────────────────────────────────
print("=" * 55)
print("  MEB Otonom Araç — Kamera Test Modu")
print("=" * 55)

try:
    modeli_yukle(MODEL_YOLU)
except Exception as e:
    print(f"[HATA] Model yüklenemedi: {e}")
    print("  best.pt dosyasının proje klasöründe olduğundan emin ol.")
    sys.exit(1)

# ── Kamera aç ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(KAMERA_INDEX)
if not cap.isOpened():
    print(f"[HATA] Kamera {KAMERA_INDEX} açılamadı. INDEX'i değiştirmeyi dene.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_GENISLIK)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_YUKSEKLIK)

print(f"[KAMERA] {FRAME_GENISLIK}×{FRAME_YUKSEKLIK} @ index={KAMERA_INDEX}")
print("[BİLGİ]  Çıkmak için 'q' tuşuna bas.\n")

# ── Durum takibi ───────────────────────────────────────────────────────────
son_tabela_zamani = 0       # Cooldown için son tepki zamanı
son_tespit_bilgisi = ""     # Ekrana yazılacak son durum mesajı
bekleme_bitis = 0           # 5 saniyelik beklemenin bitiş zamanı

frame_sayaci = 0
fps_olcer   = time.time()

# ── Ana döngü ──────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("[HATA] Kameradan kare alınamadı.")
        break

    frame = cv2.resize(frame, (FRAME_GENISLIK, FRAME_YUKSEKLIK))
    su_an = time.time()

    # Bekleme modundayken kareyi oku ama YOLOv8 çalıştırma (CPU/GPU tasarrufu)
    if su_an < bekleme_bitis:
        kalan = bekleme_bitis - su_an
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_GENISLIK, FRAME_YUKSEKLIK),
                      (0, 0, 180), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        cv2.putText(frame, f"BEKLEME: {kalan:.1f} sn",
                    (20, FRAME_YUKSEKLIK // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("MEB Otonom — Kamera Testi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ── YOLOv8 tespiti ────────────────────────────────────────────────────
    tespitler = tahmin_yap(frame)

    # Kutuları çiz
    for ad, conf, (x1, y1, x2, y2) in tespitler:
        kutu_rengi = RENK["uyari"] if ad in DURMA_SINIFLARI else RENK["default"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), kutu_rengi, 2)
        etiket = f"{ad}  {conf:.0%}"
        cv2.putText(frame, etiket, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, kutu_rengi, 2)

    # ── Görev simülasyonu (cooldown korumalı) ──────────────────────────────
    cooldown_bitti = (su_an - son_tabela_zamani) > TABELA_COOLDOWN

    if tespitler and cooldown_bitti:
        siniflar = [ad for ad, _, _ in tespitler]

        # Dur/bekleme tabelaları — Görev 2 & 4
        dur_tespit = [ad for ad in siniflar if ad in DURMA_SINIFLARI]
        if dur_tespit:
            mesaj = f"[SİMÜLASYON] MOTORLAR DURDU — 5 SANİYE BEKLENİYOR  ({dur_tespit[0]})"
            print(f"\n{'─'*60}")
            print(mesaj)
            print(f"{'─'*60}\n")
            son_tabela_zamani = su_an
            bekleme_bitis     = su_an + 5      # Ekran bekleme sayacı

        # Yeşil ışık — Görev 1 başlangıcı
        elif "yesil" in siniflar:
            print("[SİMÜLASYON] YEŞİL IŞIK — MOTORLAR BAŞLADI (hız=40)")
            son_tabela_zamani = su_an

        # Hız tümseği — Görev 3
        elif "HizTumseği" in siniflar:
            print("[SİMÜLASYON] HIZ TÜMSEĞİ — HIZ DÜŞÜRÜLDÜ (hız=20)")
            son_tabela_zamani = su_an

        # Sollama serbest — Görev 5
        elif "SollamaSerbest" in siniflar:
            print("[SİMÜLASYON] SOLLAMA SERBEST — SOL ŞERİDE GEÇİLİYOR")
            son_tabela_zamani = su_an

        # Çıkmaz yol — Görev 6
        elif "CikmazYol" in siniflar:
            print("[SİMÜLASYON] ÇIKMAZ YOL — SAĞA DÖNÜŞ YAPILIYOR")
            son_tabela_zamani = su_an

        # Park tabelası — Görev 7 öncesi
        elif "Park" in siniflar:
            print("[SİMÜLASYON] PARK TABELASI — KIRMIZI ALAN ARANACAK")
            son_tabela_zamani = su_an

        # Kırmızı park alanı — Görev 7 bitiş
        elif "KirmiziPark" in siniflar:
            print("[SİMÜLASYON] KIRMIZI PARK ALANI — MOTORLAR DURDU. YARIŞMA BİTTİ.")
            son_tabela_zamani = su_an

    # ── HUD (bilgi katmanı) ────────────────────────────────────────────────
    # FPS hesapla
    frame_sayaci += 1
    if su_an - fps_olcer >= 1.0:
        fps = frame_sayaci / (su_an - fps_olcer)
        fps_olcer    = su_an
        frame_sayaci = 0
    else:
        fps = 0

    # Tespit sayısı
    tespit_metni = (
        "Tespit yok" if not tespitler
        else " | ".join(f"{ad}({conf:.0%})" for ad, conf, _ in tespitler)
    )

    # Cooldown göstergesi
    kalan_cd = max(0.0, TABELA_COOLDOWN - (su_an - son_tabela_zamani))
    cd_metni  = f"Cooldown: {kalan_cd:.0f}s" if kalan_cd > 0 else "Hazir"

    cv2.rectangle(frame, (0, 0), (FRAME_GENISLIK, 28), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS:{fps:4.1f}  {cd_metni}  |  {tespit_metni}",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RENK["bilgi"], 1)

    cv2.imshow("MEB Otonom — Kamera Testi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Temizlik ───────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("\n[BİTİŞ] Test sonlandırıldı.")
