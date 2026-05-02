"""
ros2_beyin_yayin.py
MEB Otonom Araç — ROS2 Yayıncı (Publisher) Node'u

Bu node kameradan gelen görüntüyü YOLOv8 ile işler ve aldığı kararları
ROS2 topic'leri üzerinden yayınlar. Motora doğrudan bağlantı YOKTUR;
ayrı bir abone (subscriber) node motor sürücüsünü yönetir.

Jetson üzerinde çalıştırmak için:
    source /opt/ros/humble/setup.bash        # ROS2 ortamını aktifleştir
    python3 ros2_beyin_yayin.py

──────────────────────────────────────────────────────────────────────────────
  TOPIC MİMARİSİ  (Alara için: hangi topic'ler var, ne işe yarıyor?)
──────────────────────────────────────────────────────────────────────────────

  BU NODE YAYINLAR (Publisher):
  ┌─────────────────┬───────────────────┬──────────────────────────────────┐
  │ Topic adı       │ Mesaj tipi        │ Ne zaman / ne içeriyor?          │
  ├─────────────────┼───────────────────┼──────────────────────────────────┤
  │ /arac_komut     │ std_msgs/String   │ Tabela/ışık kararları (aşağıda) │
  │ /serit_sapma    │ std_msgs/String   │ Her karede "SAPMA:XX" (şerit)   │
  └─────────────────┴───────────────────┴──────────────────────────────────┘

  /arac_komut mesaj listesi:
    "YESIL_ISIK"        → Trafik ışığı yeşile döndü, yarışma başlıyor
    "DUR"               → Yaya/hemzemin tabelası görüldü, motorlar dur
    "BEKLEME_BITTI"     → 5 saniyelik bekleme bitti, tekrar hareket et
    "HIZ:20"            → Hız tümseği — yavaş geç
    "HIZ:40"            → Normal hız — şerit takibine devam
    "SOLLAMA"           → Sollama serbest tabelası — sol şerit manevrası
    "SAGA_DON"          → Çıkmaz yol tabelası — sağa dönüş yap
    "PARK_TABELASI"     → Park tabelası görüldü — kırmızı alan aranıyor
    "PARK_ET"           → Kırmızı park alanı bulundu — park et, bitiş

  BAŞKA NODE'LARIN DİNLEMESİ GEREKEN TOPIC'LER:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Motor Sürücü Node'u şunları dinlemeli:                             │
  │    /arac_komut  → komuta göre motorları kontrol et                  │
  │    /serit_sapma → sapma değerine göre direksiyonu ayarla            │
  │                                                                      │
  │  İsteğe bağlı — Debug / Kayıt Node'u:                              │
  │    /arac_komut  → aldığı komutları log dosyasına yaz                │
  └──────────────────────────────────────────────────────────────────────┘
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Proje klasörünü import path'e ekle (Jetson'da farklı dizinden çalışırsa)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nesne_beyni import (modeli_yukle, tahmin_yap,
                          yesil_isik_var_mi, dur_komutu_var_mi, DURMA_SINIFLARI)
from serit_beyni  import otonom_beyin

# ── Ayarlar ────────────────────────────────────────────────────────────────
MODEL_YOLU      = "best.pt"
KAMERA_INDEX    = 0          # Jetson CSI kamera için /dev/video0 → index 0
FRAME_GENISLIK  = 640
FRAME_YUKSEKLIK = 480
TABELA_COOLDOWN = 12         # Aynı tabelayı tekrar işleme almadan bekleme (sn)
GORUNTU_GOSTER  = False      # Jetson'da monitör yoksa False yap (headless mod)


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 Node sınıfı
# ══════════════════════════════════════════════════════════════════════════════

class AracBeyniNode(Node):
    """
    Kamera → YOLOv8 → ROS2 karar yayıncısı.

    İç durum makinesi (state machine):
        ISIK_BEKLE  : Trafik ışığı yeşile dönene kadar bekle (yarışma başlamadı)
        NORMAL      : Şerit takibi + tabela tespiti aktif
        DUR_BEKLE   : Yaya/hemzemin tabelası nedeniyle 5 sn bekleniyor
    """

    ISIK_BEKLE = "ISIK_BEKLE"
    NORMAL     = "NORMAL"
    DUR_BEKLE  = "DUR_BEKLE"

    def __init__(self):
        super().__init__("arac_beyni")

        # ── Publisher'lar ──────────────────────────────────────────────────
        # queue_size=10: ağ gecikmesi olursa en fazla 10 mesaj bekletilir
        self.komut_yay = self.create_publisher(String, "/arac_komut",  10)
        self.sapma_yay = self.create_publisher(String, "/serit_sapma", 10)

        # ── Kamera ────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(KAMERA_INDEX)
        if not self.cap.isOpened():
            self.get_logger().fatal(f"Kamera index={KAMERA_INDEX} açılamadı!")
            raise RuntimeError("Kamera açılamadı")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_GENISLIK)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_YUKSEKLIK)

        # ── YOLOv8 modeli ─────────────────────────────────────────────────
        modeli_yukle(MODEL_YOLU)

        # ── Durum makinesi değişkenleri ───────────────────────────────────
        self.durum             = self.ISIK_BEKLE  # Başlangıç durumu
        self.son_tabela_zamani = 0.0              # Cooldown başlangıcı
        self.bekleme_bitis     = 0.0              # DUR_BEKLE bitiş zamanı

        # ── Ana döngü timer'ı: ~30 FPS ────────────────────────────────────
        # ROS2'de time.sleep() KULLANILMAZ; bunun yerine timer callback kullanılır.
        # 0.033 sn = ~30 kare/sn
        self.timer = self.create_timer(0.033, self._kare_isle)

        self.get_logger().info("━━━ AracBeyniNode başlatıldı ━━━")
        self.get_logger().info(f"  Yayın: /arac_komut | /serit_sapma")
        self.get_logger().info(f"  Durum: {self.durum}")

    # ── Yayın yardımcısı ──────────────────────────────────────────────────

    def _yayinla(self, topic_yay, mesaj: str) -> None:
        """Seçilen topic'e mesaj yayınlar ve terminale loglar."""
        msg = String()
        msg.data = mesaj
        topic_yay.publish(msg)
        # ROS2 logger: Jetson'da `ros2 topic echo /arac_komut` ile izlenebilir
        self.get_logger().info(f"[YAY] {topic_yay.topic_name} ← '{mesaj}'")

    # ── Ana callback: her ~33ms'de bir çağrılır ───────────────────────────

    def _kare_isle(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Kameradan kare alınamadı, atlanıyor.")
            return

        frame  = cv2.resize(frame, (FRAME_GENISLIK, FRAME_YUKSEKLIK))
        su_an  = time.time()

        # ── DURUM: DUR_BEKLE ──────────────────────────────────────────────
        # 5 saniyelik zorunlu bekleme süresi dolana kadar hiçbir şey yapma.
        # (Kılavuz 3.4.2 ve 3.4.4: araç en az 5 sn beklemeli)
        if self.durum == self.DUR_BEKLE:
            if su_an < self.bekleme_bitis:
                kalan = self.bekleme_bitis - su_an
                self.get_logger().info(f"[BEKLEME] {kalan:.1f} sn kaldı...", throttle_duration_sec=1)
                if GORUNTU_GOSTER:
                    self._bekle_ekrani(frame, kalan)
                return
            else:
                # Bekleme bitti → normal sürüşe dön
                self._yayinla(self.komut_yay, "BEKLEME_BITTI")
                self._yayinla(self.komut_yay, "HIZ:40")
                self.durum = self.NORMAL

        # ── YOLOv8 tespiti (her karede çalışır) ──────────────────────────
        tespitler  = tahmin_yap(frame)
        siniflar   = [ad for ad, _, _ in tespitler]

        # ── DURUM: ISIK_BEKLE ─────────────────────────────────────────────
        # Yarışma resmi olarak başlamadan önce motor komutu gönderilmez.
        # (Kılavuz 3.4.1: trafik ışığı yeşile dönünce ≤3 sn içinde hareket)
        if self.durum == self.ISIK_BEKLE:
            if yesil_isik_var_mi(tespitler, frame):
                self.get_logger().info("YEŞİL IŞIK ALGILANDI — Yarışma başlıyor!")
                self._yayinla(self.komut_yay, "YESIL_ISIK")
                self._yayinla(self.komut_yay, "HIZ:40")
                self.durum = self.NORMAL
            # Işık bekleme ekranı (isteğe bağlı)
            if GORUNTU_GOSTER:
                cv2.imshow("Beyin Node", frame)
                cv2.waitKey(1)
            return   # Aşağıdaki normal sürüş koduna geçme

        # ── DURUM: NORMAL — tabela mantığı ───────────────────────────────
        cooldown_ok = (su_an - self.son_tabela_zamani) > TABELA_COOLDOWN

        if siniflar and cooldown_ok:

            # 1) YAYA GEÇİDİ veya HEMZEMİN GEÇİT (Görev 2 & 4)
            #    dur_komutu_var_mi() IsikTabelasi tespitini HSV ile doğrular:
            #    gerçek trafik ışığı → yoksay, uyarı tabelası → dur komutu ver
            if dur_komutu_var_mi(tespitler, frame):
                self.get_logger().warn(
                    "DUR komutu — 5 sn bekleniyor"
                )
                self._yayinla(self.komut_yay, "DUR")
                # Durumu güncelle; bekleme timer'ı kare callback'te kontrol edilir
                self.durum         = self.DUR_BEKLE
                self.bekleme_bitis = su_an + 5
                self.son_tabela_zamani = su_an
                return

            # 2) HIZ TÜMSEĞİ (Görev 3)
            #    Durma yok; sadece hızı düşür, tümsek geçilince normal hıza dön.
            elif "HizTumseği" in siniflar:
                self._yayinla(self.komut_yay, "HIZ:20")
                self.son_tabela_zamani = su_an

            # 3) SOLLAMA SERBEST (Görev 5)
            #    Sol şerite geç, orange aracı sol şeritten geç, geri dön.
            #    Manevranın detayı motor sürücü node'unda uygulanır.
            elif "SollamaSerbest" in siniflar:
                self._yayinla(self.komut_yay, "SOLLAMA")
                self.son_tabela_zamani = su_an

            # 4) ÇIKMAZ YOL (Görev 6)
            #    Çıkmaz yola GİRME; sağa dön ve parkura devam et.
            elif "CikmazYol" in siniflar:
                self._yayinla(self.komut_yay, "SAGA_DON")
                self.son_tabela_zamani = su_an

            # 5) PARK TABELASI (Görev 7 — ön uyarı)
            #    Kırmızı park alanı yakında; motoru ayarla, alan beklensin.
            elif "Park" in siniflar:
                self._yayinla(self.komut_yay, "PARK_TABELASI")
                self.son_tabela_zamani = su_an

            # 6) KIRMIZI PARK ALANI (Görev 7 — bitiş)
            #    Araç kırmızı alana girdi; dur, yarışma tamamlandı.
            elif "KirmiziPark" in siniflar:
                self._yayinla(self.komut_yay, "PARK_ET")
                self.son_tabela_zamani = su_an

        # ── Şerit takibi: sapma değerini /serit_sapma'ya yayınla ─────────
        # otonom_beyin() perspektif dönüşümü + histogram ile merkez sapmasını verir.
        # Motor sürücü node'u bu değeri alıp direksiyonu (servo/ESC) ayarlar.
        try:
            _, sapma = otonom_beyin(frame)
            self._yayinla(self.sapma_yay, f"SAPMA:{sapma}")
        except Exception as hata:
            self.get_logger().warn(f"Şerit takibi hatası: {hata}", throttle_duration_sec=5)

        # ── Görüntü (Jetson'da GORUNTU_GOSTER=True ise) ───────────────────
        if GORUNTU_GOSTER:
            for ad, conf, (x1, y1, x2, y2) in tespitler:
                renk = (0, 0, 255) if ad in DURMA_SINIFLARI else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(frame, f"{ad} {conf:.0%}", (x1, max(y1-6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1)
            cv2.imshow("Beyin Node", frame)
            cv2.waitKey(1)

    # ── Bekleme ekranı (yalnızca GORUNTU_GOSTER=True ise) ─────────────────

    def _bekle_ekrani(self, frame, kalan: float) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_GENISLIK, FRAME_YUKSEKLIK),
                      (0, 0, 160), -1)
        goruntu = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        cv2.putText(goruntu, f"BEKLEME: {kalan:.1f} sn",
                    (20, FRAME_YUKSEKLIK // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 2)
        cv2.imshow("Beyin Node", goruntu)
        cv2.waitKey(1)

    # ── Node kapanırken temizlik ───────────────────────────────────────────

    def destroy_node(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("AracBeyniNode kapatıldı.")
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
#  Giriş noktası — Jetson'da: python3 ros2_beyin_yayin.py
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    # ROS2 iletişim katmanını başlat
    rclpy.init(args=args)

    try:
        node = AracBeyniNode()
    except RuntimeError as e:
        print(f"[HATA] Node başlatılamadı: {e}")
        rclpy.shutdown()
        sys.exit(1)

    try:
        # spin(): Node'u ayakta tutar; timer callback'lerini ve mesajları işler.
        # Ctrl+C veya rclpy.shutdown() çağrılana kadar buradan çıkmaz.
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[BİTİŞ] Ctrl+C alındı, node kapatılıyor...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
