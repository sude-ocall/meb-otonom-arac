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
    "IsikTabelasi":     (0.20, 0.55),  # Trafik ışığı direği — dar uzun (sıkılaştırıldı:
                                       # 0.65 üst sınır kırmızı yuvarlak/kare tabelaları
                                       # IsikTabelasi olarak geçirebiliyordu)
    "yesil":            (0.25, 0.85),
    "kirmizi":          (0.25, 0.85),
    "sari":             (0.25, 0.85),
    "soladonulmez":     (0.80, 1.25),  # Yuvarlak tabela — DOĞRU ÇALIŞIYOR, dokunma
    "sagadonulmez":     (0.80, 1.25),  #                  DOĞRU ÇALIŞIYOR, dokunma
    "girilmez":         (0.80, 1.25),  # Yuvarlak — Görev 6 (çıkmaz yol yerine)
    "park":             (0.70, 1.40),  # Mavi kare (P harfi); kırmızı kenar varsa elenir
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

# Sınıf → renk YASAĞI: bu sınıfta ilgili renk OLMAMALI. Pozitif renk doğrulaması
# yetmediğinde "yanlış sınıf"ları elemek için. Örnek: model 'parkyasak'ı 'park'
# olarak verirse mavi de var (içeride), kırmızı da (kenarda). Pozitif sadece mavi
# kontrolüyle ayırt edilemez. Kırmızı YOK kontrolüyle parkyasak elenir.
RENK_DOGRULAMA_NEGATIF: dict[str, tuple[str, float]] = {
    "park": ("kirmizi", 0.05),   # Park'ta kırmızı %5'ten fazla varsa parkyasak'tır → ele
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

        # Negatif renk kontrolü — yanlış sınıflandırılmış tabelayı ele
        # (ör. parkyasak'ı park olarak geçiren model çıktısı)
        neg = RENK_DOGRULAMA_NEGATIF.get(cls_adi)
        if neg is not None:
            yasak_renk, esik_oran = neg
            if yasak_renk == "kirmizi" and kirmizi_var_mi(
                    frame, x1, y1, x2, y2, min_oran=esik_oran):
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
    YOLOv8 'yesil' tespitinin üstüne sıkı HSV onayı.

    Önceki sürüm sarı ışığı yeşil olarak geçiriyordu çünkü:
      • Eşik 'yesil > sari' yeterliydi (1.0× baskınlık)
      • Sarı (H 20-45) ve yeşil (H 50-90) aralıkları çok yakındı, geçiş tonları
        yanlış kategoriye düşebiliyordu.

    Yeni mantık:
      1. Tüm bbox'a bak (alt yarı yerine) — küçük ışık balonlarında yarıya bölmek
         anlam taşımıyor; tüm bbox renklerin gerçek dağılımını verir.
      2. Sarı ile yeşil arasında 10°'lik H gap: sarı 20-35, yeşil 45-90.
         Aradaki belirsiz tonlar (35-45) hiçbir kategoriye girmez → güvenli ret.
      3. Yeşil hem sarıdan hem kırmızıdan EN AZ 1.5× baskın olmalı.
    """
    for ad, conf, (x1, y1, x2, y2) in tespitler:
        if ad != "yesil":
            continue

        if frame is None:
            return True

        bolge = frame[max(0, y1):y2, max(0, x1):x2]
        if bolge.size == 0:
            continue

        hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)

        yesil_say = cv2.countNonZero(cv2.inRange(
            hsv, np.array([45, 80, 80]), np.array([90, 255, 255])
        ))
        sari_say = cv2.countNonZero(cv2.inRange(
            hsv, np.array([20, 80, 80]), np.array([35, 255, 255])
        ))
        kirmizi_say = cv2.countNonZero(
            cv2.inRange(hsv, np.array([0,   80, 80]), np.array([10,  255, 255])) |
            cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        )

        min_piksel = max(15, int(bolge.shape[0] * bolge.shape[1] * 0.04))
        if (yesil_say >= min_piksel
                and yesil_say > sari_say * 1.5
                and yesil_say > kirmizi_say * 1.5):
            return True

    return False


def _isik_tabelasi_mi_yoksa_uyari_mi(frame, x1, y1, x2, y2) -> bool:
    """
    Model 'IsikTabelasi' dediğinde:
      True  → bu bir kırmızı UYARI TABELASI (dur/girilmez gibi) → DUR komutu ver
      False → bu gerçek bir TRAFİK IŞIĞI direği → DUR komutu verme

    Eski mantık 'kirmizi_orani > 0.12 → uyari' diyordu, ama trafik ışığında
    kırmızı yandığında bu da yüksek çıkıyor → kırmızı yanan trafik ışığını da
    'uyarı tabelası' sanıp DUR yayınlıyordu. Yeni mantık:

      • Trafik ışığı direğinin AYIRT EDİCİ özelliği: dikey eksende EN AZ İKİ
        FARKLI renk bandı bulundurur (kırmızı + sarı/yeşil cam bölgeleri).
      • Kırmızı uyarı tabelası TEK renkli (sadece kırmızı, başka renk anlamlı
        oranda yok).

    Bu yüzden 3 dikey bölgeye bakıp renk dağılımını analiz ederiz.
    """
    bolge = frame[max(0, y1):y2, max(0, x1):x2]
    if bolge.size == 0 or bolge.shape[0] < 9:
        return True   # bbox çok küçük/bozuk → güvenli tarafta DUR komutu

    hsv    = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)
    toplam = bolge.shape[0] * bolge.shape[1]

    # HSV aralıkları yesil_isik_var_mi ile aynı (tutarlı sınıflandırma)
    yesil_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([45, 80, 80]), np.array([90, 255, 255]))
    )
    sari_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([20, 80, 80]), np.array([35, 255, 255]))
    )
    kirmizi_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([0,   80, 80]), np.array([10,  255, 255])) |
        cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    )

    kir_o = kirmizi_say / toplam
    sar_o = sari_say    / toplam
    yes_o = yesil_say   / toplam

    # Trafik ışığı direği: kırmızı + (sarı VEYA yeşil) ikisi birden anlamlı
    # oranda → bu bir trafik ışığı, üzerinde yanan ışığa göre karar verilir,
    # tabela değil. DUR komutu yayınlama.
    if kir_o >= 0.04 and (sar_o >= 0.04 or yes_o >= 0.04):
        return False

    # Kırmızı baskın ama sarı/yeşil yok → bu kırmızı bir uyarı tabelası.
    # (dur tabelası, kırmızı kenarlı uyarı vb.) → DUR komutu yayınla.
    if kir_o >= 0.10:
        return True

    # Net karar verilemiyor — şüpheli durum. Yarış güvenliği için DUR.
    return True


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


# ── Sollama yasağı tabelası ─────────────────────────────────────────────────
# DİKKAT: Bu sınıf mevcut YOLO modelinde (best.pt) YOK.
# Model sınıf listesi: IsikTabelasi, Tunel, YayaGecidi, dur, durak, forward,
#   girilmez, iki_yonlu_trafik, ileri_sag_mecburi, ileri_sol_mecburi, ilerisag,
#   ilerisol, kavsak, kirmizi, park, parkyasak, saga_birlesim, sagadonulmez,
#   sagdangidiniz, sari, sola_birlesim, soladonulmez, soldangidiniz, turnleft,
#   turnright, yesil
#
# Sollama yasağı tabelasını gerçekten tespit edebilmek için iki seçenek var:
#   A) Modeli yeniden eğitmek (önerilen):
#      Eğitim setine 'sollama_yasagi' sınıfı için etiketli görüntüler ekleyip
#      `yolo train` ile yeni best.pt üretmek.
#   B) HSV/şekil tabanlı detektör (gorev_dedektor.py'a):
#      Kırmızı kenarlı yuvarlak içinde iki araç silüeti aramak. Yanlış pozitif
#      oranı yüksek olur (girilmez ile karışır), yarışta riskli.
#
# Şu an için boş bir placeholder bırakıyoruz; model güncellenince bu fonksiyon
# diğer 'var_mi' fonksiyonları gibi sınıf adıyla doğrudan çalışacak.
def sollama_yasagi_var_mi(tespitler: list) -> bool:
    """
    YOLO model güncellenince 'sollama_yasagi' sınıf adı kontrol edilecek.
    Model şu an bu sınıfı içermediğinden daima False döner.
    """
    return sinif_var_mi(tespitler, "sollama_yasagi")


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
