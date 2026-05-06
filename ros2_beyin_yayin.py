"""
ros2_beyin_yayin.py
MEB Otonom Araç — ROS2 Yayıncı (Publisher) Node'u

Kameradan gelen görüntüyü YOLOv8 ile işler ve aldığı kararları
ROS2 topic'leri üzerinden yayınlar. Motora doğrudan bağlantı YOKTUR.

Son review iyileştirmeleri:
  • Kamera capture ayrı thread'de  → ROS2 executor bloklanmaz
  • Temporal kararlılık filtresi   → tek-kare false positive'lere geçit yok
  • Heartbeat publisher (5Hz)      → motor watchdog'u beyin ölünce frene basar
  • MultiThreadedExecutor          → timer + heartbeat paralel çalışır
  • QoS: komut Reliable, sapma BestEffort/depth=1 (latest-wins)
  • Magic string'ler komutlar.py'den (typo riski yok)
  • Jetson CSI için isteğe bağlı GStreamer pipeline

Çalıştırma:
    source /opt/ros/humble/setup.bash
    python3 ros2_beyin_yayin.py

──────────────────────────────────────────────────────────────────────────────
  TOPIC MİMARİSİ  (Alara için)
──────────────────────────────────────────────────────────────────────────────

  YAYINLAR:
    /arac_komut    String   Yüksek seviye karar komutları (DUR, HIZ:40, ...)
    /serit_sapma   String   "SAPMA:XX" — şerit takibi sapması
    /beyin_kalp    String   "KALP" — saniyede 5 kez heartbeat (watchdog için)

  /arac_komut mesajları için → komutlar.py içindeki Komut sınıfı
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import threading
import queue
from collections import deque

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from std_msgs.msg import String

# Proje klasörünü import path'e ekle (Jetson'da farklı dizinden çalışırsa)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nesne_beyni import (modeli_yukle, tahmin_yap,
                          yesil_isik_var_mi, dur_komutu_var_mi,
                          cikmaz_yol_var_mi, park_tabelasi_var_mi,
                          kirmizi_park_alani_bul, park_alani_yonu,
                          YAYA_YAKIN_MIN_ALAN)
from serit_beyni import otonom_beyin
from gorev_dedektor import (hiz_tumsek_var_mi, hemzemin_var_mi,
                             turuncu_arac_var_mi)
from komutlar import (Komut, Topic,
                      HIZ_NORMAL, HIZ_YAVAS, HIZ_PARK, HIZ_PARK_SON, KALP_HZ)
from buton import (buton_kullanilabilir_mi, buton_hazirla,
                   buton_basildi_mi_bekle, buton_temizle)

# ── Ayarlar ────────────────────────────────────────────────────────────────
KAMERA_INDEX     = 0
FRAME_GENISLIK   = 640
FRAME_YUKSEKLIK  = 480
TABELA_COOLDOWN  = 12         # Aynı tabelayı tekrar işleme almadan bekleme (sn)
GORUNTU_GOSTER   = False      # Jetson'da monitör yoksa False (headless)

# Temporal kararlılık: son N frame'in en az M'sinde tabela görüldüyse kabul et
KARARLI_PENCERE  = 5
KARARLI_ESIK     = 3

# Pi 4'te YOLO+HSV pipeline'ı 200-300 ms sürebiliyor. Her karede çalıştırırsak
# şerit takibi ve heartbeat geç kalır. Tespit/HSV-görev kontrollerini her N
# karede bir yapıyoruz; şerit takibi (otonom_beyin) her karede çalışmaya devam.
# 3 = saniyede ~10 tespit, 100 ms ortalama gecikme — yarış için yeterli.
TESPIT_PERIYOT   = 3

# Kılavuz 4.4 — azami yarış süresi 4 dakika (240 sn).
# Brain bu süreyi geçince DUR yayınlar; motor watchdog zaten aktiftir.
YARIS_SURESI_SN  = 240

# Görev 3 (hız tümseği): yakınlık eşiği — frame yüksekliğinin yüzdesi.
# 0.85 → tümseğin alt kenarı frame'in alt %15'ine girince yavaşla.
TUMSEK_YAKINLIK_ESIGI  = 0.85
TUMSEK_YAVAS_SURE_SN   = 2.5    # bu kadar yavaş gidip normale dön

# Görev 4 (hemzemin geçit): 30 cm kala dur (kılavuz 3.4.4)
HEMZEMIN_YAKINLIK_ESIGI = 0.85

# Görev 5 (sollama): turuncu araç bbox alanı bu eşiği geçince sollama tetiklenir
# (~50-80 cm mesafe). Çok erken tetiklerse yükselt, geç tetiklerse düşür.
SOLLAMA_TETIK_ALAN     = 6000
# Cooldown motor manevra süresinden (5 sn, ros2_motor_dinleyici.SURE_SOLLAMA_FAZ_C)
# UZUN olmalı — yoksa manevra ortasında brain ikinci kez SOLLAMA yayınlar.
SOLLAMA_COOLDOWN       = 12

# ── Kamera pipeline seçimi ────────────────────────────────────────────────
# Üç mod destekleniyor:
#   1. USB kamera (V4L2)              → KAMERA_GSTREAMER = None  (default)
#   2. Raspberry Pi Camera (libcamera)→ KAMERA_GSTREAMER = _PI_PIPELINE
#   3. Jetson CSI (nvarguscamerasrc)  → KAMERA_GSTREAMER = _JETSON_PIPELINE

# Pi 4 için libcamera tabanlı GStreamer pipeline (Pi Camera v2/v3/HQ).
# NOT: Pi OS'ta `gstreamer1.0-libcamera` paketi kurulu olmalı:
#   sudo apt install gstreamer1.0-libcamera
_PI_PIPELINE = (
    "libcamerasrc ! "
    f"video/x-raw, width={FRAME_GENISLIK}, height={FRAME_YUKSEKLIK}, "
    "framerate=30/1, format=BGR ! "
    "appsink drop=1 max-buffers=1"
)

# Jetson CSI kamera (Xavier/Nano) — Pi4'te ÇALIŞMAZ, NVIDIA'ya özel
_JETSON_PIPELINE = (
    "nvarguscamerasrc ! "
    f"video/x-raw(memory:NVMM), width={FRAME_GENISLIK*2}, height={FRAME_YUKSEKLIK*2}, "
    "framerate=30/1 ! nvvidconv ! "
    f"video/x-raw, width={FRAME_GENISLIK}, height={FRAME_YUKSEKLIK}, "
    "format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1 max-buffers=1"
)

# Donanımına göre seç: Pi 4 + Pi Camera ise _PI_PIPELINE, USB ise None
KAMERA_GSTREAMER: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
class AracBeyniNode(Node):
    """Kamera → YOLOv8 → ROS2 karar yayıncısı."""

    ISIK_BEKLE = "ISIK_BEKLE"
    NORMAL     = "NORMAL"
    DUR_BEKLE  = "DUR_BEKLE"
    PARK_ARAMA = "PARK_ARAMA"   # Park tabelası görüldü, HSV ile kırmızı alan aranıyor

    def __init__(self):
        super().__init__("arac_beyni")

        # ── QoS profilleri ─────────────────────────────────────────────────
        # Komut: Reliable + KeepLast — DUR komutu kaybolursa felaket olur
        komut_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        # Sapma + heartbeat: BestEffort + depth=1 — en yenisi geçerli, eskiler atılır
        sensor_qos = qos_profile_sensor_data

        # ── Publisher'lar ──────────────────────────────────────────────────
        self.komut_yay = self.create_publisher(String, Topic.ARAC_KOMUT,  komut_qos)
        self.sapma_yay = self.create_publisher(String, Topic.SERIT_SAPMA, sensor_qos)
        self.kalp_yay  = self.create_publisher(String, Topic.KALP_ATIS,   sensor_qos)

        # ── Kamera ────────────────────────────────────────────────────────
        if KAMERA_GSTREAMER:
            self.cap = cv2.VideoCapture(KAMERA_GSTREAMER, cv2.CAP_GSTREAMER)
            self.get_logger().info("Kamera: GStreamer pipeline (Jetson CSI)")
        else:
            self.cap = cv2.VideoCapture(KAMERA_INDEX)
            # MJPEG codec: USB kameralar default YUYV gönderir, ham 640×480
            # YUYV ≈ 600 KB/kare → Pi USB-2'de 30 FPS sığmaz, driver kareleri
            # biriktirip gecikme yaratır. MJPEG ≈ 30-80 KB/kare → tıkanma yok.
            self.cap.set(cv2.CAP_PROP_FOURCC,
                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_GENISLIK)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_YUKSEKLIK)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # Driver kuyruğunu 1 kareye indir → eski/birikmiş kareler atılır,
            # gecikme ~0 olur. V4L2 driver'ı destekliyorsa etkili olur.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.get_logger().info(
                f"Kamera: V4L2 index={KAMERA_INDEX} (MJPEG, 30 FPS, buf=1)"
            )

        if not self.cap.isOpened():
            self.get_logger().fatal("Kamera açılamadı!")
            raise RuntimeError("Kamera açılamadı")

        # ── YOLOv8 modeli (best.engine varsa otomatik onu seçer) ──────────
        modeli_yukle()

        # ── Durum makinesi değişkenleri ───────────────────────────────────
        self.durum             = self.ISIK_BEKLE
        self.son_tabela_zamani = 0.0
        self.bekleme_bitis     = 0.0
        self.yaris_baslangic   = 0.0   # Yeşil ışıkla doldurulur, bitişte log için
        self.yaris_bitti       = False # 240 sn doldu mu?

        # Görev 3: hız tümseğinde yavaş gitme zamanlayıcısı
        # 0.0 → tümsek modu pasif, >0 → bu zamanda HIZ_NORMAL'e dön
        self._tumsek_donus_zamani = 0.0

        # Görev 5: sollama cooldown (komut motora gittikten sonra tetiklenmesin)
        self._son_sollama_zamani  = 0.0
        # Sollama aktif → turuncu kaybolduğunda erken bitiş tetiklenecek
        self._sollama_aktif       = False
        self._sollama_bitti_yayinlandi = False

        # Görev 7: park yaklaşmada hız zaten düşürüldüyse tekrar yayınlama
        self._park_son_hiza_dusuruldu = False

        # ── Temporal kararlılık penceresi ─────────────────────────────────
        # Son N frame'deki sınıf set'lerini tutar. _kararli() oy çokluğuna bakar.
        self._tabela_gecmis: deque[set[str]] = deque(maxlen=KARARLI_PENCERE)

        # ── Frame skip sayacı (YOLO/HSV her TESPIT_PERIYOT karede çalışır) ─
        self._kare_sayisi   = 0
        self._son_tespitler: list = []

        # ── Frame queue: capture thread → ana callback ────────────────────
        self._frame_kuyrugu: queue.Queue = queue.Queue(maxsize=1)
        self._kapaniyor      = threading.Event()
        self._kamera_thread  = threading.Thread(target=self._kamera_dongusu, daemon=True)
        self._kamera_thread.start()

        # ── Callback group: timer'lar paralel ─────────────────────────────
        cb_group = ReentrantCallbackGroup()
        self.timer      = self.create_timer(0.033, self._kare_isle, callback_group=cb_group)
        self.kalp_timer = self.create_timer(1.0 / KALP_HZ, self._kalp_at, callback_group=cb_group)

        self.get_logger().info("━━━ AracBeyniNode başlatıldı ━━━")
        self.get_logger().info(f"  Durum: {self.durum}")

    # ── Kamera thread'i: en yeni frame'i tutar, eskileri at ────────────────
    def _kamera_dongusu(self) -> None:
        while not self._kapaniyor.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            if self._frame_kuyrugu.full():
                try:
                    self._frame_kuyrugu.get_nowait()
                except queue.Empty:
                    pass
            self._frame_kuyrugu.put(frame)

    # ── Heartbeat: motor watchdog'u bunu izler ─────────────────────────────
    def _kalp_at(self) -> None:
        msg = String()
        msg.data = Komut.KALP
        self.kalp_yay.publish(msg)

    # ── Yayın yardımcısı ──────────────────────────────────────────────────
    def _yayinla(self, topic_yay, mesaj: str) -> None:
        m = String()
        m.data = mesaj
        topic_yay.publish(m)
        self.get_logger().info(f"{self._t()}[YAY] {topic_yay.topic_name} ← '{mesaj}'")

    # ── Yarış-zamanı log prefix ───────────────────────────────────────────
    def _t(self) -> str:
        """
        '[T+45.3s] ' formatında prefix.
        Yarış başlamadıysa '[T-pre] ' döner.
        Kritik log'larda kullan; throttle'lı log'larda gereksiz.
        """
        if self.yaris_baslangic <= 0:
            return "[T-pre] "
        gecen = time.time() - self.yaris_baslangic
        return f"[T+{gecen:5.1f}s] "

    # ── Temporal kararlılık kontrolü ──────────────────────────────────────
    def _kararli(self, sinif: str) -> bool:
        """Bu sınıf son KARARLI_PENCERE frame'in en az KARARLI_ESIK'inde göründü mü?"""
        return sum(sinif in s for s in self._tabela_gecmis) >= KARARLI_ESIK

    # ── Ana callback (33ms timer) ─────────────────────────────────────────
    def _kare_isle(self) -> None:
        try:
            frame = self._frame_kuyrugu.get_nowait()
        except queue.Empty:
            return

        su_an = time.time()

        # ── Yarış sonu: kılavuz 4.4 — 240 sn azami süre ───────────────────
        if (self.yaris_baslangic > 0 and not self.yaris_bitti and
                (su_an - self.yaris_baslangic) >= YARIS_SURESI_SN):
            gecen = su_an - self.yaris_baslangic
            self.get_logger().warn(
                f"{self._t()}4 DAKİKA DOLDU ({gecen:.1f} sn) — yarış sonlandırılıyor"
            )
            self._yayinla(self.komut_yay, Komut.DUR)
            self.yaris_bitti = True
            self._son_dur_tekrar = su_an
        # Defansif: yarış bittiyse her 1 sn'de bir DUR'u tekrar yayınla
        # (RELIABLE QoS olmasına rağmen extra güvenlik — heartbeat aktif kalır)
        if self.yaris_bitti:
            if su_an - getattr(self, "_son_dur_tekrar", 0) >= 1.0:
                self._yayinla(self.komut_yay, Komut.DUR)
                self._son_dur_tekrar = su_an
            return

        # ── Görev 3: Hız tümseği yavaşlatma süresi dolduysa normale dön ───
        if (self._tumsek_donus_zamani > 0 and
                su_an >= self._tumsek_donus_zamani):
            self.get_logger().info("Hız tümseği geçildi — normal hıza dönülüyor")
            self._yayinla(self.komut_yay, Komut.hiz(HIZ_NORMAL))
            self._tumsek_donus_zamani = 0.0

        # ── DUR_BEKLE: 5 sn zorunlu bekleme (Kılavuz 3.4.2 / 3.4.4) ───────
        if self.durum == self.DUR_BEKLE:
            if su_an < self.bekleme_bitis:
                kalan = self.bekleme_bitis - su_an
                self.get_logger().info(
                    f"[BEKLEME] {kalan:.1f} sn", throttle_duration_sec=1
                )
                if GORUNTU_GOSTER:
                    self._bekle_ekrani(frame, kalan)
                return
            self._yayinla(self.komut_yay, Komut.BEKLEME_BITTI)
            self._yayinla(self.komut_yay, Komut.hiz(HIZ_NORMAL))
            self.durum = self.NORMAL

        # ── Frame skip: YOLO ve NORMAL-HSV detektörler her N karede 1 ─────
        # Şerit takibi (otonom_beyin) ve PARK_ARAMA HSV her karede çalışır.
        self._kare_sayisi += 1
        tespit_yap = (self._kare_sayisi % TESPIT_PERIYOT == 0)

        # ── YOLOv8 tespiti + temporal pencere güncelleme ──────────────────
        if tespit_yap:
            tespitler = tahmin_yap(frame)
            self._son_tespitler = tespitler
            siniflar  = {ad for ad, _, _ in tespitler}
            self._tabela_gecmis.append(siniflar)
        else:
            tespitler = self._son_tespitler
            siniflar  = {ad for ad, _, _ in tespitler}

        # ── ISIK_BEKLE: yeşil ışık görene kadar hareket yok ───────────────
        if self.durum == self.ISIK_BEKLE:
            if yesil_isik_var_mi(tespitler, frame):
                # Yaris baslangicini önce ayarla ki _t() doğru göstersin
                self.yaris_baslangic = su_an
                self.get_logger().info(f"{self._t()}YEŞİL IŞIK — yarışma başlıyor!")
                self._yayinla(self.komut_yay, Komut.YESIL_ISIK)
                self._yayinla(self.komut_yay, Komut.hiz(HIZ_NORMAL))
                self.durum = self.NORMAL
            if GORUNTU_GOSTER:
                cv2.imshow("Beyin", frame); cv2.waitKey(1)
            return

        # ── PARK_ARAMA: HSV ile yerdeki kırmızı park alanını ara ──────────
        # Görev 7 — model 'KirmiziPark' sınıfını tanımıyor, bu yüzden
        # park tabelası gördükten sonra HSV ile zemindeki üç renkten
        # SADECE kırmızıyı seçeriz; mavi/yeşil alanları yok sayar.
        if self.durum == self.PARK_ARAMA:
            var, merkez, alan, icinde_mi = kirmizi_park_alani_bul(frame)
            if var:
                yon = park_alani_yonu(frame, merkez)

                # icinde_mi → kırmızı zemin frame'in dibine kadar uzanıyor
                # → araç kırmızının üstünde, dur (kılavuz 3.4.7)
                if icinde_mi:
                    bitirme = su_an - self.yaris_baslangic
                    bonus = max(0, int(YARIS_SURESI_SN - bitirme))
                    self.get_logger().warn(
                        f"{self._t()}PARK TAMAMLANDI — bitirme={bitirme:.1f}s, "
                        f"süre katsayısı bonusu={bonus}"
                    )
                    self._yayinla(self.komut_yay, Komut.PARK_ET)
                    return

                # Kademeli yaklaşma: kırmızı blob merkezi frame'in alt %35'ine
                # girdiyse araç parka çok yakın → son yaklaşma hızına geç.
                _, cy = merkez
                if (cy > frame.shape[0] * 0.65
                        and not self._park_son_hiza_dusuruldu):
                    self.get_logger().info(
                        f"Park alanı yakın (cy={cy}) — HIZ:{HIZ_PARK_SON}"
                    )
                    self._yayinla(self.komut_yay, Komut.hiz(HIZ_PARK_SON))
                    self._park_son_hiza_dusuruldu = True

                # Yaklaşma: kırmızıya doğru direksiyonu kır.
                # return ile şerit takibinin sapmayı üzerine yazmasını engelle.
                self.get_logger().info(
                    f"Kırmızı alan bulundu — yön={yon:+d} alan={alan} cy={cy}",
                    throttle_duration_sec=1
                )
                self._yayinla(self.sapma_yay, f"SAPMA:{yon}")
                return
            # Henüz kırmızı görünmedi → şerit takibi normal sapma yayınlamaya devam etsin

        # ── NORMAL: tabela mantığı (cooldown + temporal filter) ───────────
        cooldown_ok = (su_an - self.son_tabela_zamani) > TABELA_COOLDOWN

        if cooldown_ok and siniflar:
            # 1) DUR sınıfları (Görev 2 — yaya geçidi / dur tabelası)
            # min_alan=YAYA_YAKIN_MIN_ALAN ile ~30 cm kala dur (kılavuz 3.4.2)
            if dur_komutu_var_mi(
                tespitler, frame, min_alan=YAYA_YAKIN_MIN_ALAN
            ) and any(
                self._kararli(s) for s in
                ("YayaGecidi", "dur", "IsikTabelasi")
            ):
                self.get_logger().warn(f"{self._t()}YAYA GEÇİDİ (yakın) — 5 sn dur")
                self._yayinla(self.komut_yay, Komut.DUR)
                self.durum = self.DUR_BEKLE
                self.bekleme_bitis = su_an + 5
                self.son_tabela_zamani = su_an
                return

            # 2) Çıkmaz yol — modelde 'CikmazYol' yok, 'girilmez' kullanılır
            elif cikmaz_yol_var_mi(tespitler) and self._kararli("girilmez"):
                self.get_logger().warn(f"{self._t()}GİRİLMEZ — sağa dönüş")
                self._yayinla(self.komut_yay, Komut.SAGA_DON)
                self.son_tabela_zamani = su_an

            # 3) Park tabelası (Görev 7 — kırmızı arama moduna geç)
            elif park_tabelasi_var_mi(tespitler) and self._kararli("park"):
                self.get_logger().info(f"{self._t()}PARK TABELASI — kırmızı alan aranıyor")
                self._yayinla(self.komut_yay, Komut.PARK_TABELASI)
                self.durum = self.PARK_ARAMA
                self.son_tabela_zamani = su_an

        # ──────────────────────────────────────────────────────────────────
        # HSV tabanlı görevler — model'de sınıfı yok, paralel kontrol
        # PARK_ARAMA durumunda tetiklenmesin (park bölgesinde başka şey yok)
        # tespit_yap koşulu: bu 3 HSV pipeline her karede ~15-30 ms tutuyordu
        # ──────────────────────────────────────────────────────────────────
        if self.durum == self.NORMAL and tespit_yap:

            # ── Görev 4: Hemzemin geçit (kılavuz 3.4.4 — 30 cm + 5 sn) ────
            # Yaya geçidi ile aynı bekleme protokolü; cooldown 'tabela' ile paylaşılır
            if cooldown_ok:
                var_hz, yakin_hz = hemzemin_var_mi(frame)
                if var_hz and yakin_hz >= HEMZEMIN_YAKINLIK_ESIGI:
                    self.get_logger().warn(
                        f"{self._t()}HEMZEMİN GEÇİT (yakın={yakin_hz:.2f}) — 5 sn dur"
                    )
                    self._yayinla(self.komut_yay, Komut.DUR)
                    self.durum = self.DUR_BEKLE
                    self.bekleme_bitis = su_an + 5
                    self.son_tabela_zamani = su_an
                    return

            # ── Görev 3: Hız tümseği (kılavuz 3.4.3 — yavaş geç) ──────────
            # Cooldown'dan bağımsız — tümsek üzerinde tekrar tetiklemek zarar vermez
            if self._tumsek_donus_zamani == 0.0:
                var_t, yakin_t = hiz_tumsek_var_mi(frame)
                if var_t and yakin_t >= TUMSEK_YAKINLIK_ESIGI:
                    self.get_logger().warn(
                        f"{self._t()}HIZ TÜMSEĞİ (yakın={yakin_t:.2f}) — yavaşla"
                    )
                    self._yayinla(self.komut_yay, Komut.hiz(HIZ_YAVAS))
                    self._tumsek_donus_zamani = su_an + TUMSEK_YAVAS_SURE_SN

            # ── Görev 5: Turuncu araç → sollama (kılavuz 3.4.5) ────────────
            # Kılavuz 3.1: turuncu araç yalnızca sollama serbest bölgede olur,
            # bu yüzden ayrı bir tabela kontrolü gerekmez.
            var_o, alan_o, _ = turuncu_arac_var_mi(frame)

            # Akıllı erken çıkış: sollama aktifken turuncu artık görünmüyorsa
            # motor'a SOLLAMA_BITTI yayınla → manevra Faz C'ye atlasın.
            # Manevra başlangıcından sonra en az 1 sn geçmiş olmalı (sahte erken
            # çıkışı engelle: kameraya turuncu daha bbox eşiğine girmemiş olabilir).
            if self._sollama_aktif and not self._sollama_bitti_yayinlandi:
                gecen = su_an - self._son_sollama_zamani
                if gecen > 1.0 and not var_o:
                    self.get_logger().info(
                        f"Turuncu araç kayboldu ({gecen:.1f}sn) — SOLLAMA_BITTI"
                    )
                    self._yayinla(self.komut_yay, Komut.SOLLAMA_BITTI)
                    self._sollama_bitti_yayinlandi = True

            sollama_cooldown_ok = (
                (su_an - self._son_sollama_zamani) > SOLLAMA_COOLDOWN
            )
            if sollama_cooldown_ok and not self._sollama_aktif:
                if var_o and alan_o >= SOLLAMA_TETIK_ALAN:
                    self.get_logger().warn(
                        f"TURUNCU ARAÇ (alan={alan_o}) — SOLLAMA tetikleniyor"
                    )
                    self._yayinla(self.komut_yay, Komut.SOLLAMA)
                    self._son_sollama_zamani = su_an
                    self._sollama_aktif = True
                    self._sollama_bitti_yayinlandi = False

            # Cooldown sona erdiyse sollama state sıfırla
            if (self._sollama_aktif and
                    (su_an - self._son_sollama_zamani) >= SOLLAMA_COOLDOWN):
                self._sollama_aktif = False
                self._sollama_bitti_yayinlandi = False

        # ── Şerit takibi: sapma değerini sürekli yayınla ──────────────────
        try:
            _, sapma = otonom_beyin(frame)
            self._yayinla(self.sapma_yay, f"SAPMA:{sapma}")
        except Exception as hata:
            self.get_logger().warn(
                f"Şerit takibi hatası: {hata}", throttle_duration_sec=5
            )

        # ── Görüntü (debug) ──────────────────────────────────────────────
        if GORUNTU_GOSTER:
            for ad, conf, (x1, y1, x2, y2) in tespitler:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{ad} {conf:.0%}", (x1, max(y1-6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Beyin", frame); cv2.waitKey(1)

    # ── Bekleme ekranı (yalnızca GORUNTU_GOSTER=True ise) ─────────────────
    def _bekle_ekrani(self, frame, kalan: float) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 160), -1)
        goruntu = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        cv2.putText(goruntu, f"BEKLEME: {kalan:.1f} sn",
                    (20, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 2)
        cv2.imshow("Beyin", goruntu); cv2.waitKey(1)

    # ── Node kapanırken temizlik ──────────────────────────────────────────
    def destroy_node(self) -> None:
        self._kapaniyor.set()
        self._kamera_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("AracBeyniNode kapatıldı.")
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    # ── Görev 1, kademe 0: BUTON İLE BAŞLATMA (kılavuz 3.4.1, +50 puan) ───
    # Buton donanımı yoksa otomatik olarak atlanır (test ortamı için).
    # Node hiç başlamaz → motor watchdog'u sessiz kalır → hiçbir şey hareket etmez.
    if buton_kullanilabilir_mi() and buton_hazirla():
        print("[GÖREV 1] Buton modu aktif — fiziksel butona basılması bekleniyor.")
        buton_basildi_mi_bekle()
    else:
        print("[GÖREV 1] Buton donanımı tespit edilmedi — doğrudan ROS2 başlatılıyor.")

    rclpy.init(args=args)
    try:
        node = AracBeyniNode()
    except RuntimeError as e:
        print(f"[HATA] Node başlatılamadı: {e}")
        rclpy.shutdown()
        buton_temizle()
        sys.exit(1)

    # MultiThreadedExecutor: timer + heartbeat + (varsa) gelecekte abonelikler paralel
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n[BİTİŞ] Ctrl+C alındı, node kapatılıyor...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        buton_temizle()


if __name__ == "__main__":
    main()
