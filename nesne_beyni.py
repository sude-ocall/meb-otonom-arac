"""
nesne_beyni.py
YOLOv8 tabanlı nesne/tabela/ışık tespiti.
Tüm inference bu modülden geçer; main.py davranış kararlarını verir.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from goruntu_islem import on_isle, kirmizi_var_mi, mavi_var_mi, sari_var_mi

# ── Ayarlar ────────────────────────────────────────────────────────────────
# TensorRT engine varsa önce onu kullan (Jetson'da 3-4× hızlı)
# Export komutu (Jetson terminalinde çalıştır):
#   yolo export model=best.pt format=engine half=True device=0 imgsz=416
MODEL_YOLU   = "best.engine" if os.path.exists("best.engine") else "best.pt"

GUVEN_ESIGI  = 0.35         # Confidence threshold — 0.35 dengeli başlangıç noktası
                            # Çok yanlış tespit olursa 0.45'e çek
                            # Hâlâ "tespit yok" diyorsa 0.25'e düşür

# Minimum bounding-box alanı (piksel²)
# Bu değerin altındaki kutular gürültü — atılır.
# 400x300 frame'de 20×20 px = 400, 25×25 = 625
MIN_KUTU_ALANI = 500

# Sınıf bazlı beklenen en-boy oranı (genişlik / yükseklik) aralıkları.
# Bu aralık dışına düşen kutular şekil uyumsuzluğu nedeniyle atılır.
# → Kırmızı tişört gibi geniş nesneler "dur" (kare tabela) yerine geçemez.
# NOT: Modelindeki gerçek sınıf adlarına göre düzenle.
ASPEKT_ORANI: dict[str, tuple[float, float]] = {
    "dur":           (0.75, 1.30),  # Sekizgen — yaklaşık kare
    "YayaGecidi":    (0.65, 1.55),  # Kare tabela
    "HemzeminGecit": (0.65, 1.55),
    "IsikTabelasi":  (0.25, 0.65),  # Trafik ışığı direği — dar, uzun
    "yesil":         (0.25, 0.65),
    "kirmizi":       (0.25, 0.65),
    "sari":          (0.25, 0.65),
    "HizTumseği":    (0.65, 1.55),
    "SolaDonulmez":  (0.80, 1.25),  # Yuvarlak tabela
    "SagaDonulmez":  (0.80, 1.25),
    "SollamaSerbest":(0.65, 1.55),
    "CikmazYol":     (0.65, 1.55),
    "Park":          (0.65, 1.55),
    "KirmiziPark":   (1.80, 10.0),  # Yerdeki kırmızı alan — yatay uzun
}

# Hangi sınıflar için renk doğrulaması zorunlu?
# Sınıf → hangi renk fonksiyonu çalışsın
# False-positive önleme: model "dur" dese de bbox'ta kırmızı yoksa at.
RENK_DOGRULAMA: dict[str, str] = {
    "dur":           "kirmizi",
    "YayaGecidi":    "kirmizi",
    "HemzeminGecit": "kirmizi",
    "SolaDonulmez":  "kirmizi",
    "SagaDonulmez":  "kirmizi",
    "SollamaSerbest":"mavi",
    "HizTumseği":    "sari",
}

# Birbirine çok benzeyen tabela çiftleri için sınıf bazlı yüksek eşik.
# Model bu çiftleri düşük güvenle karıştırıyor; bu eşikler düşük-güvenli
# yanlış tespiti keser.
# ÖNEMLİ: test_tespit.py başlarken sınıf adlarını terminale yazar.
#          Modelindeki gerçek adlar farklıysa burayı güncelle.
SINIF_ESIGI: dict[str, float] = {
    "SolaDonulmez": 0.65,   # Sola/sağa dönülmez çifti çok benzer
    "SagaDonulmez": 0.65,
    "SolaGec":      0.60,
    "SagaGec":      0.60,
}

# Hangi sınıf adları "dur ve 5 sn bekle" davranışını tetikler?
# Modelindeki isimlere göre buraya ekle / çıkar.
DURMA_SINIFLARI = {
    "YayaGecidi",       # Görev 2 — yaya geçidi tabelası
    "HemzeminGecit",    # Görev 4 — hemzemin geçit tabelası
    "dur",              # genel dur tabelası
    "IsikTabelasi",     # Model bazen yaya/hemzemin tabelasını bununla karıştırıyor;
                        # dur_komutu_var_mi() gerçek ışık mı tabela mı diye HSV ile ayırt eder
}

# ── Model tek seferlik yüklenir ────────────────────────────────────────────
_model: YOLO | None = None


def modeli_yukle(yol: str = MODEL_YOLU) -> None:
    """best.pt dosyasını belleğe yükler. main.py başında bir kez çağrılır."""
    global _model
    _model = YOLO(yol)
    print(f"[MODEL] '{yol}' yüklendi. Sınıflar: {list(_model.names.values())}")


# ── Çekirdek inference ─────────────────────────────────────────────────────

def tahmin_yap(frame) -> list[tuple[str, float, tuple]]:
    """
    Verilen BGR karesi üzerinde çıkarım yapar.
    Döner: [(sinif_adi, confidence, (x1,y1,x2,y2)), ...]

    Filtre zinciri (her tespit sırayla geçer):
      1. Genel güven eşiği (GUVEN_ESIGI)
      2. Sınıf bazlı yüksek eşik (SINIF_ESIGI — benzer tabela çiftleri)
      3. Minimum kutu alanı (MIN_KUTU_ALANI — gürültü eler)
      4. Aspect-ratio kontrolü (ASPEKT_ORANI — yanlış şekilli kutular eler)
      5. HSV renk doğrulaması (RENK_DOGRULAMA — rengi uymayan kutular eler)
    """
    if _model is None:
        raise RuntimeError("modeli_yukle() henüz çağrılmadı.")

    isle_frame = on_isle(frame, clahe=True)   # CLAHE parlama azaltma
    results    = _model(isle_frame, verbose=False)[0]
    tespitler  = []

    for box in results.boxes:
        conf    = float(box.conf[0])           # type: ignore[index]
        cls_id  = int(box.cls[0])              # type: ignore[index]
        cls_adi = results.names[cls_id]

        # 1. Genel güven eşiği
        if conf < GUVEN_ESIGI:
            continue

        # 2. Karıştırılan tabela çiftleri için sınıf bazlı yüksek eşik
        if conf < SINIF_ESIGI.get(cls_adi, GUVEN_ESIGI):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])  # type: ignore[index]
        w, h_box = x2 - x1, y2 - y1

        # 3. Çok küçük kutular gürültü — at
        if w * h_box < MIN_KUTU_ALANI:
            continue

        # 4. Aspect-ratio — şekli uymayan kutu false-positive'dir
        if h_box > 0:
            aspekt = w / h_box
            ar_min, ar_max = ASPEKT_ORANI.get(cls_adi, (0.15, 8.0))
            if not (ar_min <= aspekt <= ar_max):
                continue

        # 5. HSV renk doğrulaması — model "dur" dedi ama kırmızı piksel yok → at
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
    """tespitler içinde verilen sınıf(lar)dan biri var mı?"""
    if isinstance(siniflar, str):
        siniflar = {siniflar}
    return any(ad in siniflar for ad, _, _ in tespitler)


def yesil_isik_var_mi(tespitler: list, frame=None) -> bool:
    """
    Trafik ışığı gerçekten yeşil mi?

    İki kademeli doğrulama:
      1. YOLOv8 "yesil" sınıfını güvenle tespit etmeli.
      2. frame verilirse bbox'ın ALT yarısında HSV yeşil piksel kontrolü yapılır.
         (Trafik ışığında yeşil lamba en altta bulunur.)

    Neden gerekli?
      Model bazen kırmızı lambayı yeşil olarak yanlış sınıflandırabilir.
      HSV onayı bu yanlış tetiklenmeyi engeller.
    """
    for ad, conf, (x1, y1, x2, y2) in tespitler:
        if ad != "yesil":
            continue

        # frame yoksa model kararına güven (eski davranış)
        if frame is None:
            return True

        # Trafik ışığında yeşil lamba alt kısımda → bbox'ın alt %45'ini al
        h_bbox = y2 - y1
        alt_y1 = y1 + int(h_bbox * 0.55)
        bolge  = frame[max(0, alt_y1):y2, max(0, x1):x2]

        if bolge.size == 0:
            return True   # Kırpma başarısızsa modele güven

        hsv = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)

        # Yeşil: H=50-90  (H=40-49 sarıyla çakışır; 50'den başlatarak ayrım sağlanır)
        yesil_mask = cv2.inRange(
            hsv, np.array([50, 80, 80]), np.array([90, 255, 255])
        )
        # Sarı: H=20-45  (trafik ışığı sarısı buraya düşer → yeşil sayılmamalı)
        sari_mask = cv2.inRange(
            hsv, np.array([20, 80, 80]), np.array([45, 255, 255])
        )
        # Kırmızı: HSV'de iki ayrı aralık
        kirmizi_mask = (
            cv2.inRange(hsv, np.array([0,   60, 80]), np.array([10,  255, 255])) |
            cv2.inRange(hsv, np.array([170, 60, 80]), np.array([180, 255, 255]))
        )

        yesil_say   = cv2.countNonZero(yesil_mask)
        sari_say    = cv2.countNonZero(sari_mask)
        kirmizi_say = cv2.countNonZero(kirmizi_mask)

        # Onay: yeterli yeşil piksel VE kırmızıdan fazla VE sarıdan da fazla
        # → sarı ışık artık yanlışlıkla "yeşil" saymaz
        min_piksel = max(10, int(bolge.shape[0] * bolge.shape[1] * 0.05))
        if yesil_say >= min_piksel and yesil_say > kirmizi_say and yesil_say > sari_say:
            return True

    return False


def _isik_tabelasi_mi_yoksa_uyari_mi(frame, x1, y1, x2, y2) -> bool:
    """
    Model 'IsikTabelasi' dediğinde gerçekten uyarı tabelası mı kontrol eder.

    İki kademeli karar:
      1. Kırmızı baskın (>%12) → kesinlikle uyarı tabelası; arka plandaki yeşil
         (çimen, ağaç) ne kadar yüksek olursa olsun dur komutu ver.
      2. Kırmızı az ama yeşil+sarı da az (<8%) → yine tabela kabul et.
      3. Yeşil+sarı yeterince yüksek (≥%8) → gerçek trafik ışığı.

    Döner: True  → uyarı tabelası (dur komutu ver)
           False → gerçek trafik ışığı (dur komutu verme)
    """
    bolge = frame[max(0, y1):y2, max(0, x1):x2]
    if bolge.size == 0:
        return True  # Kırpma başarısızsa tabela kabul et

    hsv    = cv2.cvtColor(bolge, cv2.COLOR_BGR2HSV)
    toplam = bolge.shape[0] * bolge.shape[1]

    yesil_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([40, 60, 80]), np.array([90, 255, 255]))
    )
    sari_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([20, 60, 80]), np.array([38, 255, 255]))
    )
    # HSV'de kırmızı iki ayrı aralıkta yer alır
    kirmizi_say = cv2.countNonZero(
        cv2.inRange(hsv, np.array([0,   60, 80]), np.array([10,  255, 255])) |
        cv2.inRange(hsv, np.array([170, 60, 80]), np.array([180, 255, 255]))
    )

    kirmizi_orani = kirmizi_say / toplam
    isik_orani    = (yesil_say + sari_say) / toplam

    # Kırmızı baskın → arka plandaki yeşilden bağımsız olarak tabela
    if kirmizi_orani > 0.12:
        return True

    # Yeşil+sarı yetersiz → tabela
    return isik_orani < 0.08


def dur_komutu_var_mi(tespitler: list, frame=None) -> bool:
    """
    5 saniyelik bekleme gerektiren tabela görüldü mü?

    'IsikTabelasi' tespitinde ek HSV doğrulaması yapılır:
      - Gerçek trafik ışığı → yoksay (sadece yesil/kirmizi ile ilgilenilir)
      - Uyarı tabelası (yaya/hemzemin ile karışmış) → dur komutu ver
    """
    for ad, _, (x1, y1, x2, y2) in tespitler:
        # Kesin dur sınıfları — doğrudan kabul
        if ad in {"YayaGecidi", "HemzeminGecit", "dur"}:
            return True

        # IsikTabelasi: gerçek ışık mı tabela mı?
        if ad == "IsikTabelasi":
            if frame is None:
                return True   # frame yoksa güvenli taraf: dur komutu ver
            if _isik_tabelasi_mi_yoksa_uyari_mi(frame, x1, y1, x2, y2):
                return True   # Uyarı tabelası — dur

    return False


def hiz_tumseği_var_mi(tespitler: list) -> bool:
    """Hız tümseği uyarı tabelası (Görev 3)."""
    return sinif_var_mi(tespitler, "HizTumseği")


def sollama_serbest_var_mi(tespitler: list) -> bool:
    """Sollama serbest bölge tabelası (Görev 5)."""
    return sinif_var_mi(tespitler, "SollamaSerbest")


def cikmaz_yol_var_mi(tespitler: list) -> bool:
    """Çıkmaz yol tabelası (Görev 6)."""
    return sinif_var_mi(tespitler, "CikmazYol")


def park_tabelasi_var_mi(tespitler: list) -> bool:
    """Park alanı tabelası (Görev 7 öncesi uyarı)."""
    return sinif_var_mi(tespitler, "Park")


def kirmizi_park_alani_var_mi(tespitler: list) -> bool:
    """Yerdeki kırmızı park alanı görüldü mü? (Görev 7 — bitiş)"""
    return sinif_var_mi(tespitler, "KirmiziPark")


# ── Görselleştirme ─────────────────────────────────────────────────────────

def gorsele_ciz(frame, tespitler: list):
    """Tespit kutularını ve etiketleri kare üzerine çizer."""
    for ad, conf, (x1, y1, x2, y2) in tespitler:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{ad} {conf:.2f}",
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return frame
