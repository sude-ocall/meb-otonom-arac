"""
ros2_motor_dinleyici.py
MEB Otonom Araç — ROS2 Motor Sürücü Subscriber Node'u (DİFERANSİYEL SÜRÜŞ)

Donanım: Raspberry Pi 4 + L298N + 4× sarı DC gearmotor + 6×AA pil (9V)
Direksiyon: YOK — sol/sağ motor hız farkıyla dönüş yapılır.

Beyin node'unun yayınladığı komutları dinler ve sol/sağ motor PWM'lerini
diferansiyel olarak ayarlar.

Çalıştırma:
    source /opt/ros/humble/setup.bash
    python3 ros2_motor_dinleyici.py

──────────────────────────────────────────────────────────────────────────────
  TOPIC MİMARİSİ  (Emir & Alara için)
──────────────────────────────────────────────────────────────────────────────

  DİNLER:
    /arac_komut    String   Yüksek seviye karar komutları
    /serit_sapma   String   "SAPMA:XX" — şerit sapma değeri (diff hıza çevrilir)
    /beyin_kalp    String   "KALP" — watchdog; 1sn gelmezse acil fren

  Komut → Diferansiyel hız çevrimi:
    DUR / PARK_ET     → sol=0,    sağ=0
    HIZ:40            → cruise=40 (sapma callback delta uygular)
    YESIL_ISIK        → cruise=40
    SOLLAMA           → 1sn sola kay (sol yavaş, sağ hızlı)
    SAGA_DON          → 1.5sn yerinde sağa dönüş (sol+, sağ-)
    PARK_TABELASI     → cruise=25 (yavaşla)

  Sapma → diferansiyel:
    sol_hız = cruise - kazanım × sapma
    sağ_hız = cruise + kazanım × sapma
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from komutlar import (Komut, Topic,
                      HIZ_NORMAL, HIZ_PARK, HIZ_DONUS, HIZ_SOLLAMA,
                      KALP_TIMEOUT_SN)


# ═══════════════════════════════════════════════════════════════════════════
#  L298N + 4 MOTOR DİFERANSİYEL BAĞLANTI ŞEMASI
# ═══════════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  GÜÇ HATLARI:                                                        │
#  │                                                                      │
#  │   6×AA pil (9V)  ──>  L298N VIN, GND     (motorları besler)         │
#  │   USB powerbank  ──>  Pi 4 USB-C          (BEYNI ayrı besler!)      │
#  │   Pi 4 GND      ──>  L298N GND            (ortak referans şart)     │
#  └──────────────────────────────────────────────────────────────────────┘
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  L298N → MOTOR BAĞLANTISI (DİFERANSİYEL):                           │
#  │                                                                      │
#  │   Channel A (OUT1, OUT2)  ──>  SOL ön motor + SOL arka motor        │
#  │                                  (paralel; aynı yönde döner)        │
#  │   Channel B (OUT3, OUT4)  ──>  SAĞ ön motor + SAĞ arka motor        │
#  │                                  (paralel; aynı yönde döner)        │
#  └──────────────────────────────────────────────────────────────────────┘
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  Pi 4 GPIO → L298N SİNYAL PİNLERİ:                                  │
#  │                                                                      │
#  │   GPIO 17  ──>  IN1     │ Sol motorlar yön bit-1                    │
#  │   GPIO 27  ──>  IN2     │ Sol motorlar yön bit-2                    │
#  │   GPIO 18  ──>  ENA     │ Sol motorlar PWM hız (1kHz)               │
#  │   GPIO 22  ──>  IN3     │ Sağ motorlar yön bit-1                    │
#  │   GPIO 23  ──>  IN4     │ Sağ motorlar yön bit-2                    │
#  │   GPIO 24  ──>  ENB     │ Sağ motorlar PWM hız (1kHz)               │
#  └──────────────────────────────────────────────────────────────────────┘

# TODO (Emir & Alara): araç bağlanırken aşağıdaki bloğu yorumdan çıkarın.
#
# import RPi.GPIO as GPIO
#
# PIN_IN1, PIN_IN2, PIN_ENA = 17, 27, 18    # SOL taraf motorlar
# PIN_IN3, PIN_IN4, PIN_ENB = 22, 23, 24    # SAĞ taraf motorlar
#
# GPIO.setmode(GPIO.BCM)
# GPIO.setwarnings(False)
# GPIO.setup([PIN_IN1, PIN_IN2, PIN_ENA, PIN_IN3, PIN_IN4, PIN_ENB], GPIO.OUT)
# pwm_sol = GPIO.PWM(PIN_ENA, 1000)         # 1kHz motor PWM
# pwm_sag = GPIO.PWM(PIN_ENB, 1000)
# pwm_sol.start(0); pwm_sag.start(0)


# ── Sabitler ──────────────────────────────────────────────────────────────
# PWM güvenlik tavanı: motorlar 3-6V için tasarlanmış, biz 9V/L298N veriyoruz.
# Tam %100 PWM motorları yakar; %70 ile sınırla.
PWM_MAX         = 70

# Sapma → diferansiyel çevrim katsayısı
# Aşırı salınım yaparsa 0.3'e düşür; yetersiz cevap verirse 0.5'e çıkar
SAPMA_KAZANIM   = 0.4

# Filtreleme: ani hız değişimlerini yumuşat
HIZ_EWMA        = 0.6   # Yumuşatma katsayısı (0-1, yüksek = daha yumuşak)
HIZ_MIN_FARK    = 3     # Bu PWM yüzdesi altındaki değişimleri yoksay

# Manevra (SOLLAMA, SAGA_DON) süreleri — bu süre içinde sapma görmezden gelinir
SURE_SAGA_DON   = 1.5   # saniye

# ── SOLLAMA ÜÇ FAZLI MANEVRA (kılavuz 3.4.5) ──────────────────────────────
# Tek "kay sola" yetmez; aracın turuncu aracı tamamen geçip sağ şeride
# dönmesi gerek. Faz sınırları kümülatif (manevra başlangıcından itibaren).
# Donanım üstünde son ayar yapılacak — bu değerler 35% PWM'de kabaca
# 1m yana kayma + 1m düz + 1m geri kayma için tahmin.
SURE_SOLLAMA_FAZ_A   = 1.5   # 0.0 → 1.5 sn  : sol şeride kay
SURE_SOLLAMA_FAZ_B   = 3.5   # 1.5 → 3.5 sn  : düz git, turuncu aracı geç
SURE_SOLLAMA_FAZ_C   = 5.0   # 3.5 → 5.0 sn  : sağ şeride dön

# Faz A/C'de yan kayma şiddeti (0=sadece bir motor, 1=eşit)
SOLLAMA_KAYMA_ORANI  = 0.4


def kistir(deger: float, min_deg: float, max_deg: float) -> float:
    """min/max aralığına kıstırır."""
    return max(min_deg, min(max_deg, deger))


# ═══════════════════════════════════════════════════════════════════════════
class MotorDinleyiciNode(Node):
    """Komut dinler, watchdog tutar, sol/sağ motor PWM'lerini diferansiyel ayarlar."""

    def __init__(self):
        super().__init__("motor_dinleyici")

        # ── QoS profilleri (beyin ile uyumlu) ──────────────────────────────
        komut_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        sensor_qos = qos_profile_sensor_data

        # ── Callback group: paralel callback'ler ──────────────────────────
        cb = ReentrantCallbackGroup()

        # ── Subscriber'lar ─────────────────────────────────────────────────
        self.create_subscription(
            String, Topic.ARAC_KOMUT,  self._komut_cb, komut_qos, callback_group=cb
        )
        self.create_subscription(
            String, Topic.SERIT_SAPMA, self._sapma_cb, sensor_qos, callback_group=cb
        )
        self.create_subscription(
            String, Topic.KALP_ATIS,   self._kalp_cb,  sensor_qos, callback_group=cb
        )

        # ── Durum değişkenleri ─────────────────────────────────────────────
        self._cruise_hiz    = 0       # Hedef cruise PWM (0-100, sapma'sız)
        self._sol_son_pwm   = 0.0     # Son uygulanan sol motor PWM (-100..+100)
        self._sag_son_pwm   = 0.0     # Son uygulanan sağ motor PWM (-100..+100)
        self._son_kalp      = self.get_clock().now()
        self._manevra_bitis = 0.0     # Bu zaman geçene kadar sapma yoksay

        # Sollama üç fazlı manevra başlangıcı (0 = aktif değil)
        self._sollama_baslangic = 0.0

        # ── Watchdog: 200ms'de bir heartbeat kontrolü ─────────────────────
        self.create_timer(0.2, self._watchdog, callback_group=cb)

        # ── Sollama tick: 10Hz'de fazları yürüt ──────────────────────────
        self.create_timer(0.1, self._sollama_tik, callback_group=cb)

        self.get_logger().info("━━━ MotorDinleyiciNode (DİFERANSİYEL) başlatıldı ━━━")
        self.get_logger().info("  Dinleniyor: /arac_komut | /serit_sapma | /beyin_kalp")

    # ══════════════════════════════════════════════════════════════════════
    #  DONANIM FONKSİYONLARI — TODO bloklarını gerçek GPIO koduyla doldurun
    # ══════════════════════════════════════════════════════════════════════

    def _sol_motor_set(self, pwm: float) -> None:
        """
        Sol taraf motorlarına işaretli PWM uygular.
        pwm: -PWM_MAX..+PWM_MAX  (negatif = geri, pozitif = ileri, 0 = serbest)

        TODO (Emir): Aşağıdaki dummy yerine gerçek GPIO komutu:

            pwm = max(-PWM_MAX, min(PWM_MAX, pwm))   # güvenlik kıstırma
            if pwm > 0:                                # İleri
                GPIO.output(PIN_IN1, GPIO.HIGH)
                GPIO.output(PIN_IN2, GPIO.LOW)
                pwm_sol.ChangeDutyCycle(pwm)
            elif pwm < 0:                              # Geri
                GPIO.output(PIN_IN1, GPIO.LOW)
                GPIO.output(PIN_IN2, GPIO.HIGH)
                pwm_sol.ChangeDutyCycle(abs(pwm))
            else:                                      # Serbest (coast)
                GPIO.output(PIN_IN1, GPIO.LOW)
                GPIO.output(PIN_IN2, GPIO.LOW)
                pwm_sol.ChangeDutyCycle(0)
        """
        self.get_logger().debug(f"[SOL]  {pwm:+6.1f}%")

    def _sag_motor_set(self, pwm: float) -> None:
        """
        Sağ taraf motorlarına işaretli PWM uygular.
        pwm: -PWM_MAX..+PWM_MAX  (negatif = geri, pozitif = ileri)

        TODO (Emir): Aşağıdaki dummy yerine gerçek GPIO komutu:

            pwm = max(-PWM_MAX, min(PWM_MAX, pwm))
            if pwm > 0:
                GPIO.output(PIN_IN3, GPIO.HIGH)
                GPIO.output(PIN_IN4, GPIO.LOW)
                pwm_sag.ChangeDutyCycle(pwm)
            elif pwm < 0:
                GPIO.output(PIN_IN3, GPIO.LOW)
                GPIO.output(PIN_IN4, GPIO.HIGH)
                pwm_sag.ChangeDutyCycle(abs(pwm))
            else:
                GPIO.output(PIN_IN3, GPIO.LOW)
                GPIO.output(PIN_IN4, GPIO.LOW)
                pwm_sag.ChangeDutyCycle(0)
        """
        self.get_logger().debug(f"[SAĞ]  {pwm:+6.1f}%")

    def _fren_yap(self) -> None:
        """
        Tüm motorları durdurur (kısa devre fren modu).
        Sollama manevrası varsa onu da iptal eder (race condition önler).

        TODO (Emir): Aşağıdaki dummy yerine GPIO fren komutu:

            GPIO.output(PIN_IN1, GPIO.HIGH); GPIO.output(PIN_IN2, GPIO.HIGH)
            GPIO.output(PIN_IN3, GPIO.HIGH); GPIO.output(PIN_IN4, GPIO.HIGH)
            pwm_sol.ChangeDutyCycle(0); pwm_sag.ChangeDutyCycle(0)
        """
        self.get_logger().info("[FREN] motorlar durduruldu")
        self._sol_son_pwm = 0.0
        self._sag_son_pwm = 0.0
        self._sollama_baslangic = 0.0

    # ══════════════════════════════════════════════════════════════════════
    #  Diferansiyel hız uygulama yardımcısı (filtreleme + kıstırma)
    # ══════════════════════════════════════════════════════════════════════

    def _diff_uygula(self, sol_hedef: float, sag_hedef: float) -> None:
        """Hedef sol/sağ PWM'lerini EWMA filtresi + min-fark eşiği ile uygular."""
        # Güvenlik tavanı
        sol_hedef = kistir(sol_hedef, -PWM_MAX, PWM_MAX)
        sag_hedef = kistir(sag_hedef, -PWM_MAX, PWM_MAX)

        # EWMA: ani değişimleri yumuşat
        sol_yeni = HIZ_EWMA * self._sol_son_pwm + (1 - HIZ_EWMA) * sol_hedef
        sag_yeni = HIZ_EWMA * self._sag_son_pwm + (1 - HIZ_EWMA) * sag_hedef

        # Min-fark: küçük titreşimlere komut gönderme
        if (abs(sol_yeni - self._sol_son_pwm) >= HIZ_MIN_FARK or
            abs(sag_yeni - self._sag_son_pwm) >= HIZ_MIN_FARK):
            self._sol_son_pwm = sol_yeni
            self._sag_son_pwm = sag_yeni
            self._sol_motor_set(sol_yeni)
            self._sag_motor_set(sag_yeni)

    # ══════════════════════════════════════════════════════════════════════
    #  WATCHDOG: beyin sessizse acil fren
    # ══════════════════════════════════════════════════════════════════════

    def _watchdog(self) -> None:
        gecen = (self.get_clock().now() - self._son_kalp).nanoseconds / 1e9
        if gecen > KALP_TIMEOUT_SN and self._cruise_hiz != 0:
            self.get_logger().error(
                f"BEYIN BAĞLANTISI YOK ({gecen:.1f}sn) — ACİL FREN!"
            )
            self._cruise_hiz = 0
            self._fren_yap()

    # ══════════════════════════════════════════════════════════════════════
    #  SOLLAMA TICK: 10Hz — fazları yürüt
    # ══════════════════════════════════════════════════════════════════════

    def _sollama_tik(self) -> None:
        """
        Sollama manevrası fazlarını süre bazlı yürütür.
        _sollama_baslangic = 0 ise hiçbir şey yapmaz.
        """
        if self._sollama_baslangic == 0.0:
            return

        gecen = time.time() - self._sollama_baslangic

        if gecen < SURE_SOLLAMA_FAZ_A:
            # Faz A: sol şeride kay (sağ hızlı, sol yavaş)
            self._diff_uygula(
                HIZ_SOLLAMA * SOLLAMA_KAYMA_ORANI,
                HIZ_SOLLAMA,
            )
        elif gecen < SURE_SOLLAMA_FAZ_B:
            # Faz B: düz git, turuncu aracı geç
            self._diff_uygula(HIZ_SOLLAMA, HIZ_SOLLAMA)
        elif gecen < SURE_SOLLAMA_FAZ_C:
            # Faz C: sağ şeride dön (sol hızlı, sağ yavaş)
            self._diff_uygula(
                HIZ_SOLLAMA,
                HIZ_SOLLAMA * SOLLAMA_KAYMA_ORANI,
            )
        else:
            # Manevra tamamlandı — şerit takibi devraldı
            self.get_logger().info(
                f"Sollama tamamlandı ({gecen:.1f}sn) — şerit takibine geçildi"
            )
            self._sollama_baslangic = 0.0
            self._cruise_hiz = HIZ_NORMAL

    # ══════════════════════════════════════════════════════════════════════
    #  CALLBACK'ler
    # ══════════════════════════════════════════════════════════════════════

    def _kalp_cb(self, msg: String) -> None:
        """Beyin yaşıyor sinyali — watchdog'u sıfırlar."""
        self._son_kalp = self.get_clock().now()

    def _komut_cb(self, msg: String) -> None:
        """Yüksek seviye karar komutlarını işler."""
        komut = msg.data.strip()
        self.get_logger().info(f"[KOMUT] '{komut}'")

        # ── Dur / Park (yarışma sonu) ─────────────────────────────────────
        if komut in (Komut.DUR, Komut.PARK_ET):
            self._cruise_hiz = 0
            self._fren_yap()

        # ── Yeşil ışık / bekleme bitti → normal hıza geç ──────────────────
        elif komut in (Komut.YESIL_ISIK, Komut.BEKLEME_BITTI):
            self._cruise_hiz = HIZ_NORMAL
            # İlk hareket: doğrudan cruise, sapma callback delta uygular
            self._diff_uygula(HIZ_NORMAL, HIZ_NORMAL)

        # ── HIZ:XX (cruise hızı güncelleme) ──────────────────────────────
        elif komut.startswith("HIZ:"):
            hiz = Komut.hiz_parse(komut)
            if hiz is None:
                self.get_logger().error(f"Geçersiz hız: '{komut}'")
                return
            self._cruise_hiz = hiz
            # Doğrudan uygulamadan sapma callback'in delta eklemesini bekle

        # ── Sollama (Görev 5) — üç fazlı manevra ─────────────────────────
        # Faz A: sola kay   Faz B: düz git, aracı geç   Faz C: sağa dön
        # Faz mantığı _sollama_tik()'te yürür; burada sadece manevrayı tetikler.
        elif komut == Komut.SOLLAMA:
            if self._sollama_baslangic > 0:
                self.get_logger().warn("Sollama zaten devam ediyor — yok sayıldı")
                return
            self.get_logger().info("SOLLAMA başladı (3 fazlı manevra)")
            self._cruise_hiz = HIZ_SOLLAMA
            self._sollama_baslangic = time.time()
            # Tüm manevra süresince sapma callback yok say
            self._manevra_bitis = self._sollama_baslangic + SURE_SOLLAMA_FAZ_C

        # ── Sollama erken bitiş — turuncu araç kayboldu, Faz C'ye atla ─
        elif komut == Komut.SOLLAMA_BITTI:
            if self._sollama_baslangic == 0.0:
                # Manevra zaten bitmiş veya hiç başlamamış — yok say
                return
            gecen = time.time() - self._sollama_baslangic
            if gecen < SURE_SOLLAMA_FAZ_A:
                # Faz A'da iken erken çıkış güvenli değil — sola yeterince kaymadık
                self.get_logger().info(
                    "SOLLAMA_BITTI yok sayıldı (Faz A henüz tamamlanmadı)"
                )
                return
            if gecen >= SURE_SOLLAMA_FAZ_B:
                # Zaten Faz C veya sonrası — etkisi yok
                return
            # Faz B → Faz C başlangıcına atla
            atlanan = SURE_SOLLAMA_FAZ_B - gecen
            self._sollama_baslangic -= atlanan
            self._manevra_bitis = self._sollama_baslangic + SURE_SOLLAMA_FAZ_C
            self.get_logger().info(
                f"SOLLAMA_BITTI → Faz B'den C'ye atlandı ({atlanan:.1f}sn kazanıldı)"
            )

        # ── Çıkmaz yol → yerinde sağa dönüş (Görev 6) ─────────────────────
        elif komut == Komut.SAGA_DON:
            self.get_logger().warn("ÇIKMAZ YOL → yerinde sağa dönüş")
            self._cruise_hiz = HIZ_DONUS
            # Sol ileri, sağ geri → tank dönüşü (yerinde 90°)
            self._diff_uygula(+HIZ_DONUS, -HIZ_DONUS)
            self._manevra_bitis = time.time() + SURE_SAGA_DON

        # ── Park tabelası — yavaşla, kırmızı alanı bekle ──────────────────
        elif komut == Komut.PARK_TABELASI:
            self._cruise_hiz = HIZ_PARK

        else:
            self.get_logger().warn(f"Tanınmayan komut: '{komut}'")

    def _sapma_cb(self, msg: String) -> None:
        """
        Şerit sapması → diferansiyel hız delta'sı.
        Pozitif sapma → sola dönüş gerek (sol motor yavaş, sağ hızlı).
        Negatif sapma → sağa dönüş gerek (sol hızlı, sağ yavaş).
        """
        # Araç duruyor — motor sallama
        if self._cruise_hiz == 0:
            return

        # Sollama manevrası aktif — fazları _sollama_tik() yürütüyor, sapma yok say
        if self._sollama_baslangic > 0:
            return

        # Manevra süresi dolmamış (SOLLAMA / SAGA_DON aktif)
        if time.time() < self._manevra_bitis:
            return

        try:
            sapma = int(msg.data.split(":")[1])
        except (IndexError, ValueError):
            self.get_logger().warn(
                f"Geçersiz sapma: '{msg.data}'", throttle_duration_sec=5
            )
            return

        delta = SAPMA_KAZANIM * sapma
        sol_hedef = self._cruise_hiz - delta   # Sola dönüş için sol yavaşlasın
        sag_hedef = self._cruise_hiz + delta   # Sola dönüş için sağ hızlansın
        self._diff_uygula(sol_hedef, sag_hedef)

    # ══════════════════════════════════════════════════════════════════════
    #  Node kapanırken — güvenlik freni
    # ══════════════════════════════════════════════════════════════════════

    def destroy_node(self) -> None:
        self._fren_yap()
        # TODO (Emir): GPIO kullanılıyorsa: GPIO.cleanup()
        self.get_logger().info("MotorDinleyiciNode kapatıldı — motorlar durduruldu.")
        super().destroy_node()


# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = MotorDinleyiciNode()

    # 3 callback (komut + sapma + heartbeat) + watchdog timer paralel
    executor = MultiThreadedExecutor(num_threads=3)
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
