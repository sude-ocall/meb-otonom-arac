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
                          yesil_isik_var_mi, dur_komutu_var_mi)
from serit_beyni import otonom_beyin
from komutlar import (Komut, Topic,
                      HIZ_NORMAL, HIZ_YAVAS, HIZ_PARK, HIZ_DONUS, HIZ_SOLLAMA,
                      KALP_HZ)

# ── Ayarlar ────────────────────────────────────────────────────────────────
KAMERA_INDEX     = 0
FRAME_GENISLIK   = 640
FRAME_YUKSEKLIK  = 480
TABELA_COOLDOWN  = 12         # Aynı tabelayı tekrar işleme almadan bekleme (sn)
GORUNTU_GOSTER   = False      # Jetson'da monitör yoksa False (headless)

# Temporal kararlılık: son N frame'in en az M'sinde tabela görüldüyse kabul et
KARARLI_PENCERE  = 5
KARARLI_ESIK     = 3

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_GENISLIK)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_YUKSEKLIK)
            self.get_logger().info(f"Kamera: V4L2 index={KAMERA_INDEX}")

        if not self.cap.isOpened():
            self.get_logger().fatal("Kamera açılamadı!")
            raise RuntimeError("Kamera açılamadı")

        # ── YOLOv8 modeli (best.engine varsa otomatik onu seçer) ──────────
        modeli_yukle()

        # ── Durum makinesi değişkenleri ───────────────────────────────────
        self.durum             = self.ISIK_BEKLE
        self.son_tabela_zamani = 0.0
        self.bekleme_bitis     = 0.0

        # ── Temporal kararlılık penceresi ─────────────────────────────────
        # Son N frame'deki sınıf set'lerini tutar. _kararli() oy çokluğuna bakar.
        self._tabela_gecmis: deque[set[str]] = deque(maxlen=KARARLI_PENCERE)

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
        self.get_logger().info(f"[YAY] {topic_yay.topic_name} ← '{mesaj}'")

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

        # ── YOLOv8 tespiti + temporal pencere güncelleme ──────────────────
        tespitler = tahmin_yap(frame)
        siniflar  = {ad for ad, _, _ in tespitler}
        self._tabela_gecmis.append(siniflar)

        # ── ISIK_BEKLE: yeşil ışık görene kadar hareket yok ───────────────
        if self.durum == self.ISIK_BEKLE:
            if yesil_isik_var_mi(tespitler, frame):
                self.get_logger().info("YEŞİL IŞIK — yarışma başlıyor!")
                self._yayinla(self.komut_yay, Komut.YESIL_ISIK)
                self._yayinla(self.komut_yay, Komut.hiz(HIZ_NORMAL))
                self.durum = self.NORMAL
            if GORUNTU_GOSTER:
                cv2.imshow("Beyin", frame); cv2.waitKey(1)
            return

        # ── NORMAL: tabela mantığı (cooldown + temporal filter) ───────────
        cooldown_ok = (su_an - self.son_tabela_zamani) > TABELA_COOLDOWN

        if cooldown_ok and siniflar:
            # 1) DUR sınıfları (yaya/hemzemin/dur/IsikTabelasi-tabela)
            if dur_komutu_var_mi(tespitler, frame) and any(
                self._kararli(s) for s in
                ("YayaGecidi", "HemzeminGecit", "dur", "IsikTabelasi")
            ):
                self.get_logger().warn("DUR — 5 sn bekleniyor")
                self._yayinla(self.komut_yay, Komut.DUR)
                self.durum = self.DUR_BEKLE
                self.bekleme_bitis = su_an + 5
                self.son_tabela_zamani = su_an
                return

            # 2) Hız tümseği
            elif self._kararli("HizTumseği"):
                self._yayinla(self.komut_yay, Komut.hiz(HIZ_YAVAS))
                self.son_tabela_zamani = su_an

            # 3) Sollama serbest
            elif self._kararli("SollamaSerbest"):
                self._yayinla(self.komut_yay, Komut.SOLLAMA)
                self.son_tabela_zamani = su_an

            # 4) Çıkmaz yol
            elif self._kararli("CikmazYol"):
                self._yayinla(self.komut_yay, Komut.SAGA_DON)
                self.son_tabela_zamani = su_an

            # 5) Park tabelası (Görev 7 — ön uyarı)
            elif self._kararli("Park"):
                self._yayinla(self.komut_yay, Komut.PARK_TABELASI)
                self.son_tabela_zamani = su_an

            # 6) Kırmızı park alanı (Görev 7 — bitiş)
            elif self._kararli("KirmiziPark"):
                self._yayinla(self.komut_yay, Komut.PARK_ET)
                self.son_tabela_zamani = su_an

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
    rclpy.init(args=args)
    try:
        node = AracBeyniNode()
    except RuntimeError as e:
        print(f"[HATA] Node başlatılamadı: {e}")
        rclpy.shutdown()
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


if __name__ == "__main__":
    main()
