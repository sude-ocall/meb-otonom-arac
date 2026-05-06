"""
nesne_beyni.py
YOLOv8 tabanlı nesne/tabela/ışık tespiti.
Tüm inference bu modülden geçer; main.py davranış kararlarını verir.

Modeldeki gerçek sınıflar (best_ncnn_model):
  IsikTabelasi, Tunel, YayaGecidi, dur, durak, forward, girilmez,
  iki_yonlu_trafik, ileri_sag_mecburi, ileri_sol_mecburi, ilerisag,
  ilerisol, kavsak, kirmizi, park, parkyasak, saga_birlesim,
  sagadonulmez, sagdangidiniz, sari, sola_birlesim, soladonulmez,
  soldangidiniz, turnleft, turnright, yesil
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from goruntu_islem import on_isle, kirmizi_var_mi, mavi_var_mi, sari_var_mi

# ── Ayarlar ────────────────────────────────────────────────────────────────
# Model dosyası otomatik seçimi.
#   1. best_ncnn_model/   → Raspberry Pi 4 (ARM CPU, 5-8× hızlı, ÖNERİLEN)
#                          Export: yolo export model=best.pt format=ncnn imgsz=320
#   2. best.engine        → Jetson (TensorRT, CUDA gerekir)
#   3. best.pt            → Fallback (yavaş; Pi4'te 1-3 FPS)
if os.path.isdir("best_ncnn_model"):
    MODEL_YOLU = "best_ncnn_model"
elif os.path.exists("best.engine"):
    MODEL_YOLU = "best.engine"
else:
    MODEL_YOLU = "best.pt"

GUVEN_ESIGI  = 0.35         # Confidence threshold — 0.35 dengeli başlangıç
                            # Çok yanlış tespit → 0.45'e çek
                            # Hâlâ "tespit yok" → 0.25'e düşür

# 320 her zaman: Pi 4 + best.pt'de 416→320 ≈ %40 hızlanma, küçük tabela
# doğruluk kaybı pratikte ihmal edilebilir (yarış mesafelerinde).
MODEL_IMGSZ = 320

# Minimum bounding-box alanı (piksel²) — bu altı gürültü, atılır
MIN_KUTU_ALANI = 500

# Yaya geçidi tabelası 30 cm mesafede kabaca bu kadar piksel² yer kaplar.
# Kılavuz 3.4.2: yaya geçidine en fazla 30 cm kala durulmalı.
# 400×300 frame'de 13 cm tabela ≈ 80×80 px = 6400 px². Tolerans için 4500.
# Çok erken duruyorsa bu değeri yükselt (ör. 6000); duramıyorsa düşür (3000).
YAYA_YAKIN_MIN_ALAN = 4500

# Sınıf bazlı beklenen en-boy oranı (genişlik / yükseklik)
ASPEKT_ORANI: dict[str, tuple[float, float]] = {
    "dur":              (0.75, 1.30),  # Sekizgen — yaklaşık kare
    "YayaGecidi":       (0.65, 1.55),
    "IsikTabelasi":     (0.25, 0.65),  # Trafik ışığı direği — dar uzun
    "yesil":            (0.25, 0.85),
    "kirmizi":          (0.25, 0.85),
    "sari":             (0.25, 0.85),
    "soladonulmez":     (0.80, 1.25),  # Yuvarlak tabela
    "sagadonulmez":     (0.80, 1.25),
    "girilmez":         (0.80, 1.25),  # Yuvarlak — Görev 6 (çıkmaz yol yerine)
    "park":             (0.65, 1.55),
    "parkyasak":        (0.80, 1.25),
    "Tunel":            (0.65, 1.55),
    "durak":            (0.65, 1.55),
    "kavsak":           (0.65, 1.55),
    "iki_yonlu_trafik": (0.65, 1.55),
    "turnleft":         (0.80, 1.25),
    "turnright":        (0.80, 1.25),
}

# Sınıf → renk doğrulama: model bu sınıfı dediğinde HSV'de ilgili renk olmalı
RENK_DOGRULAMA: dict[str, str] = {
    "dur":           "kirmizi",
    "YayaGecidi":    "kirmizi",
    "soladonulmez":  "kirmizi",
    "sagadonulmez":  "kirmizi",
    "girilmez":      "kirmizi",
    "parkyasak":     "kirmizi",
    "park":          "mavi",   # Mavi tabela
}

# Çift halinde model karıştırıyor → daha yüksek eşik
SINIF_ESIGI: dict[str, float] = {
    "soladonulmez": 0.55,
    "sagadonulmez": 0.55,
    "soldangidiniz": 0.55,
    "sagdangidiniz": 0.55,
    "turnleft":     0.55,
    "turnright":    0.55,
}

# 5 sn dur + bekle davranışını tetikleyen sınıflar
# Görev 2 (yaya geçidi) + genel "dur" tabelası
DURMA_SINIFLARI = {
    "YayaGecidi",
    "dur",
    "IsikTabelasi",   # Bazen yaya/dur tabelasıyla karışıyor — HSV ile ayırt edilir
}

# ── Model tek seferlik yüklenir ────────────────────────────────────────────
_model: YOLO | None = None


def modeli_yukle(yol: str = MODEL_YOLU) -> None:
    """best_ncnn_model klasörünü belleğe yükler. main.py başında bir kez çağrılır."""
    global _model
    _model = YOLO(yol)
    print(f"[MODEL] '{yol}' yüklendi. Sınıflar: {list(_model.names.values())}")


# ── Çekirdek inference ─────────────────────────────────────────────────────

def tahmin_yap(frame) -> list[tuple[str, float, tuple]]:
    """
    Verilen BGR karesi üzerinde çıkarım yapar.
    Döner: [(sinif_adi, confidence, (x1,y1,x2,y2)), ...]
    """
    if _model is None:
        raise RuntimeError("modeli_yukle() henüz çağrılmadı.")

    # Adaptif CLAHE: sadece çok karanlık (<80) veya çok parlak (>180) ortamlarda
    # Sub-sample mean (8×8 grid) — full frame.mean()'den ~64× hızlı, tahmin aynı
    gri_ort = float(frame[::8, ::8].mean())
    isle_frame = on_isle(frame, clahe=True) if (gri_ort < 80 or gri_ort > 180) else frame

    results = _model(isle_frame, imgsz=MODEL_IMGSZ, verbose=False)[0]
    tespitler  = []

    for box in results.boxes:
        conf    = float(box.conf[0])
        cls_id  = int(box.cls[0])
        cls_adi = results.names[cls_id]

        if conf < GUVEN_ESIGI:
            continue

        if conf < SINIF_ESIGI.get(cls_adi, GUVEN_ESIGI):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h_box = x2 - x1, y2 - y1

        if w * h_box < MIN_KUTU_ALANI:
            continue

        if h_box > 0:
            aspekt = w / h_box
            ar_min, ar_max = ASPEKT_ORANI.get(cls_adi, (0.15, 8.0))
            if not (ar_min <= aspekt <= ar_max):
                continue

        renk = RENK_DOGRULAMA.get(cls_adi)
        if renk == "kirmizi" and not kirmizi_var_mi(frame, x1, y1, x2, y2):
            continue
        if renk == "mavi"    and not mavi_var_mi(frame, x1, y1, x2, y2):
            continue
        if renk == "sari"    and not sari_var_mi(frame, x1, y1, x2, y2):
            continue

        tespitler.append((cls_adi, conf, (x1, y1, x2, y2)))

    return tespitler


# ── Sınıf sorgulama yardımcıları ──────────────────────────────────────────

def sinif_var_mi(tespitler: list, siniflar: set | str) -> bool:
    if isinstance(siniflar, str):
        siniflar = {siniflar}
    return any(ad in siniflar for ad, _, _ in tespitler)


def yesil_isik_var_mi(tespitler: list, frame=None) -> bool:
    """
    Trafik ışığı gerçekten yeşil mi?
    YOLOv8 'yesil' tespitinin üstüne HSV onayı: bbox alt yarısında yeşil baskın olmalı.
    """
    for ad, conf, (x1, y1, x2, y2) in tespitler:
        if ad != "yesil":
            continue

        if frame is None:
            return True

        h_bbox = y2 - y1
        alt_y1 = y1 + int(h_bbox * 0.55)
        bolge  = frame[max(0, alt_y1):y2, max(0, x1):x2]

        if bolge.size == 0:
            return True

        hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)

        yesil_mask = cv2.inRange(
            hsv, np.array([50, 80, 80]), np.array([90, 255, 255])
        )
        sari_mask = cv2.inRange(
            hsv, np.array([20, 80, 80]), np.array([45, 255, 255])
        )
        kirmizi_mask = (
            cv2.inRange(hsv, np.array([0,   60, 80]), np.array([10,  255, 255])) |
            cv2.inRange(hsv, np.array([170, 60, 80]), np.array([180, 255, 255]))
        )

        yesil_say   = cv2.countNonZero(yesil_mask)
        sari_say    = cv2.countNonZero(sari_mask)
        kirmizi_say = cv2.countNonZero(kirmizi_mask)

        min_piksel = max(10, int(bolge.shape[0] * bolge.shape[1] * 0.05))
        if yesil_say >= min_piksel and yesil_say > kirmizi_say and yesil_say > sari_say:
            return True

    return False


def _isik_tabelasi_mi_yoksa_uyari_mi(frame, x1, y1, x2, y2) -> bool:
    """
    Model 'IsikTabelasi' dediğinde gerçekten uyarı tabelası mı kontrol eder.
    Kırmızı baskın → uyarı tabelası (dur). Yeşil+sarı baskın → gerçek ışık (dur verme).
    """
    bolge = frame[max(0, y1):y2, max(0, x1):x2]
    if bolge.size == 0:
        return True

    hsv    = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)
    toplam = bolge.shape[0] * bolge.shape[1]

    yesil_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([40, 60, 80]), np.array([90, 255, 255]))
    )
    sari_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([20, 60, 80]), np.array([38, 255, 255]))
    )
    kirmizi_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([0,   60, 80]), np.array([10,  255, 255])) |
        cv2.inRange(hsv, np.array([170, 60, 80]), np.array([180, 255, 255]))
    )

    kirmizi_orani = kirmizi_say / toplam
    isik_orani    = (yesil_say + sari_say) / toplam

    if kirmizi_orani > 0.12:
        return True

    return isik_orani < 0.08


def dur_komutu_var_mi(tespitler: list, frame=None, min_alan: int = 0) -> bool:
    """
    5 saniyelik bekleme gerektiren tabela görüldü mü?
    Görev 2 (yaya geçidi) için kullanılır.

    min_alan: 0 (default) → her uzaklıkta tespit kabul
              >0          → bbox alanı bu değerden büyük olmalı (yakın olmalı)
    Kılavuz 3.4.2: yaya geçidine en fazla 30 cm kala durulmalı →
    main akışında YAYA_YAKIN_MIN_ALAN ile çağrılır.
    """
    for ad, _, (x1, y1, x2, y2) in tespitler:
        if min_alan > 0:
            alan = (x2 - x1) * (y2 - y1)
            if alan < min_alan:
                continue

        if ad in {"YayaGecidi", "dur"}:
            return True

        if ad == "IsikTabelasi":
            if frame is None:
                return True
            if _isik_tabelasi_mi_yoksa_uyari_mi(frame, x1, y1, x2, y2):
                return True

    return False


def cikmaz_yol_var_mi(tespitler: list) -> bool:
    """
    Görev 6 — modelde 'CikmazYol' yok, yerine 'girilmez' tabelası kullanılır.
    Kavramsal olarak aynı: 'buraya girme, başka yöne git'.
    """
    return sinif_var_mi(tespitler, "girilmez")


def park_tabelasi_var_mi(tespitler: list) -> bool:
    """Görev 7 öncesi mavi 'park' tabelası — kırmızı alanı aramaya başla işareti."""
    return sinif_var_mi(tespitler, "park")


# ── Görev 7: Yerdeki kırmızı park alanını HSV ile bul ──────────────────────

def kirmizi_park_alani_bul(frame,
                            min_alan_orani: float = 0.03,
                            sadece_alt_yari: bool = True
                            ) -> tuple[bool, tuple[int, int] | None, int, bool]:
    """
    Kameranın gördüğü zemindeki kırmızı park alanını bulur.

    Park alanları kırmızı/mavi/yeşil olarak üç renkten oluşur. YOLOv8 modeli
    bu zemin renklerini sınıf olarak tanımıyor. Bu yüzden park modunda
    HSV maskesi ile direkt zemine bakıp en büyük kırmızı bölgeyi seçeriz —
    böylece mavi ve yeşil park alanları yanlışlıkla seçilmez.

    Parametreler:
      min_alan_orani:  Frame alanına oranla minimum kırmızı blok büyüklüğü
                       (0.03 = frame'in en az %3'ü kırmızı olmalı)
      sadece_alt_yari: Sadece görüntünün alt yarısına bak (zemin); üst yarıdaki
                       kırmızı tabelalar/şeyler etkilemesin

    Döner: (var_mi, (cx, cy) | None, alan, icinde_mi)
      - var_mi:     Kırmızı park alanı bulundu mu
      - (cx, cy):   Tüm-frame koordinatlarında kırmızı bölgenin merkezi
      - alan:       Kırmızı bölgenin piksel sayısı (büyüklük göstergesi)
      - icinde_mi:  Kırmızının en alt noktası frame'in dibine yakınsa True →
                    araç fiziksel olarak kırmızı alanın ÜSTÜNDE demektir, dur.
                    (Kılavuz 3.4.7: park alanı sınırları dışına taşmamalı)
    """
    h, w = frame.shape[:2]

    if sadece_alt_yari:
        baslama_y = h // 2
        bolge = frame[baslama_y:, :]
    else:
        baslama_y = 0
        bolge = frame

    hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)

    # HSV'de kırmızı iki ayrı aralıkta — birleşik maske
    kirmizi_mask = (
        cv2.inRange(hsv, np.array([0,   100, 70]), np.array([10,  255, 255])) |
        cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255]))
    )

    # Gürültü temizliği — küçük noktaları sil, küçük delikleri kapat
    kernel = np.ones((5, 5), np.uint8)
    kirmizi_mask = cv2.morphologyEx(kirmizi_mask, cv2.MORPH_OPEN,  kernel)
    kirmizi_mask = cv2.morphologyEx(kirmizi_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        kirmizi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False, None, 0, False

    en_buyuk = max(contours, key=cv2.contourArea)
    alan = int(cv2.contourArea(en_buyuk))

    min_alan = int(bolge.shape[0] * bolge.shape[1] * min_alan_orani)
    if alan < min_alan:
        return False, None, alan, False

    M = cv2.moments(en_buyuk)
    if M['m00'] == 0:
        return False, None, alan, False

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00']) + baslama_y    # tam frame koordinatı

    # icinde_mi: kırmızı kontur'un EN ALT y-koordinatı frame'in son %5'inde mi?
    # Eğer öyleyse araç kırmızı zemine basmış demektir → durmalıyız.
    _, _, _, h_box = cv2.boundingRect(en_buyuk)
    en_alt_y = baslama_y + (cv2.boundingRect(en_buyuk)[1] + h_box)
    icinde_mi = en_alt_y >= int(h * 0.95)

    return True, (cx, cy), alan, icinde_mi


def park_alani_yonu(frame, merkez: tuple[int, int]) -> int:
    """
    Kırmızı park alanının merkezi frame'in neresinde?
    Negatif sayı = sol, pozitif = sağ. -100..+100 arası ölçeklenir.
    Direksiyon komutu olarak doğrudan kullanılabilir.
    """
    cx, _ = merkez
    w = frame.shape[1]
    yari = w / 2
    sapma = (cx - yari) / yari   # -1..+1
    return int(max(-100, min(100, sapma * 100)))


# ── Görselleştirme ─────────────────────────────────────────────────────────

def gorsele_ciz(frame, tespitler: list):
    """Tespit kutularını ve etiketleri kare üzerine çizer."""
    for ad, conf, (x1, y1, x2, y2) in tespitler:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{ad} {conf:.2f}",
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return frame
