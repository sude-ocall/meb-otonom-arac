"""
ros2_motor_dinleyici.py
MEB Otonom Araç — ROS2 Motor Sürücü Subscriber Node'u

Bu node, ros2_beyin_yayin.py'nin yayınladığı komutları dinler ve
fiziksel motorları / servo direksiyonu kontrol eder.

Jetson / Raspberry Pi üzerinde çalıştırmak için:
    source /opt/ros/humble/setup.bash
    python3 ros2_motor_dinleyici.py

──────────────────────────────────────────────────────────────────────────────
  TOPIC MİMARİSİ  (Emir ve Alara için: bu node ne dinliyor?)
──────────────────────────────────────────────────────────────────────────────

  BU NODE DİNLER (Subscriber):
  ┌─────────────────┬─────────────────┬────────────────────────────────────┐
  │ Topic adı       │ Mesaj tipi      │ Açıklama                           │
  ├─────────────────┼─────────────────┼────────────────────────────────────┤
  │ /arac_komut     │ String          │ Yüksek seviye karar komutları      │
  │ /serit_sapma    │ String          │ Şerit takibi direksiyon sapması    │
  └─────────────────┴─────────────────┴────────────────────────────────────┘

  /arac_komut mesajları ve tetiklenen fonksiyonlar:
  ┌──────────────────┬───────────────────────────────────────────────────┐
  │ Gelen mesaj      │ Çağrılan fonksiyon(lar)                           │
  ├──────────────────┼───────────────────────────────────────────────────┤
  │ "DUR"            │ fren_yap()                                        │
  │ "PARK_ET"        │ fren_yap()          ← Yarışma bitiş durumu        │
  │ "HIZ:40"         │ motorlara_guc_ver(40)                             │
  │ "HIZ:20"         │ motorlara_guc_ver(20)  ← Hız tümseği             │
  │ "YESIL_ISIK"     │ motorlara_guc_ver(40)  ← Yarışma başlangıcı      │
  │ "BEKLEME_BITTI"  │ motorlara_guc_ver(40)  ← 5 sn bitti              │
  │ "SOLLAMA"        │ direksiyonu_cevir(sol) + motorlara_guc_ver(35)    │
  │ "SAGA_DON"       │ fren_yap() + direksiyonu_cevir(sag)               │
  │ "PARK_TABELASI"  │ motorlara_guc_ver(25)  ← Yavaş, alan aranıyor    │
  └──────────────────┴───────────────────────────────────────────────────┘

  /serit_sapma mesaj formatı: "SAPMA:XX"
      XX pozitif → araç sola kaymış → direksiyonu sola çevir
      XX negatif → araç sağa kaymış → direksiyonu sağa çevir
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ═══════════════════════════════════════════════════════════════════════════
#  DONANIM FONKSİYONLARI  (şimdilik dummy — GPIO kodu buraya yazılacak)
# ═══════════════════════════════════════════════════════════════════════════
#
#  TODO (Emir & Alara):
#  Aşağıdaki üç fonksiyonu gerçek GPIO koduyla doldurun.
#  Kullanacağınız kütüphane:
#    - Raspberry Pi  → import RPi.GPIO as GPIO   (veya gpiozero)
#    - Jetson Nano   → import Jetson.GPIO as GPIO
#
#  L298N Motor Sürücü Pin Şeması (örnek):
#  ┌──────────┬──────────┬────────────────────────────────────────────┐
#  │ L298N pin│ GPIO pin │ Açıklama                                   │
#  ├──────────┼──────────┼────────────────────────────────────────────┤
#  │ IN1      │ 17       │ Sol/ön motor yön bit-1                     │
#  │ IN2      │ 27       │ Sol/ön motor yön bit-2                     │
#  │ ENA      │ 18 (PWM) │ Sol/ön motor hız (0–100 duty cycle)        │
#  │ IN3      │ 22       │ Sağ/arka motor yön bit-1                   │
#  │ IN4      │ 23       │ Sağ/arka motor yön bit-2                   │
#  │ ENB      │ 24 (PWM) │ Sağ/arka motor hız (0–100 duty cycle)      │
#  └──────────┴──────────┴────────────────────────────────────────────┘
#
#  Direksiyon Servo Pin Şeması (örnek):
#  ┌──────────┬──────────┬────────────────────────────────────────────┐
#  │ Servo pin│ GPIO pin │ Açıklama                                   │
#  ├──────────┼──────────┼────────────────────────────────────────────┤
#  │ Sinyal   │ 12 (PWM) │ 50Hz PWM; 1ms=sol, 1.5ms=orta, 2ms=sağ   │
#  │ VCC      │ 5V       │ Güç (harici besleme önerilir)              │
#  │ GND      │ GND      │ Ortak toprak                               │
#  └──────────┴──────────┴────────────────────────────────────────────┘

# ── GPIO başlatma (TODO) ───────────────────────────────────────────────────
# TODO: Aşağıdaki satırları yorumdan çıkar ve pin numaralarını ayarla.
#
# import RPi.GPIO as GPIO          # Raspberry Pi için
# import Jetson.GPIO as GPIO       # Jetson için
#
# PIN_IN1  = 17
# PIN_IN2  = 27
# PIN_ENA  = 18   # PWM destekli pin
# PIN_IN3  = 22
# PIN_IN4  = 23
# PIN_ENB  = 24   # PWM destekli pin
# PIN_SERVO = 12  # PWM destekli pin
#
# GPIO.setmode(GPIO.BCM)
# GPIO.setup([PIN_IN1, PIN_IN2, PIN_ENA, PIN_IN3, PIN_IN4, PIN_ENB], GPIO.OUT)
# GPIO.setup(PIN_SERVO, GPIO.OUT)
# pwm_sag  = GPIO.PWM(PIN_ENA,   1000)   # 1kHz motor PWM
# pwm_sol  = GPIO.PWM(PIN_ENB,   1000)
# pwm_servo= GPIO.PWM(PIN_SERVO,   50)   # 50Hz servo PWM
# pwm_sag.start(0); pwm_sol.start(0); pwm_servo.start(7.5)  # 7.5 → orta konum

# ── Sabitleri ─────────────────────────────────────────────────────────────
SERVO_ORTA    =  0    # Düz ilerleme (derece cinsinden sıfır referans)
SERVO_MAX_SOL =  45   # Maksimum sola dönüş açısı
SERVO_MAX_SAG = -45   # Maksimum sağa dönüş açısı
HIZ_NORMAL    =  40   # Varsayılan sürüş hızı (0–100 arası PWM yüzdesi)


def motorlara_guc_ver(hiz_yuzdesi: int) -> None:
    """
    Sürüş motorlarına hız komutu gönderir.

    Parametre:
        hiz_yuzdesi: 0–100 arası tam sayı (PWM duty cycle yüzdesi)
                     0 = dur, 100 = tam hız

    TODO (Emir & Alara): Aşağıdaki dummy kodu gerçek GPIO komutuyla değiştir.
    ──────────────────────────────────────────────────────────────────────────
    # Motorları ileri yönde çalıştır (IN1=HIGH, IN2=LOW)
    GPIO.output(PIN_IN1, GPIO.HIGH)
    GPIO.output(PIN_IN2, GPIO.LOW)
    GPIO.output(PIN_IN3, GPIO.HIGH)
    GPIO.output(PIN_IN4, GPIO.LOW)
    # PWM duty cycle ile hız ayarla (0–100)
    pwm_sag.ChangeDutyCycle(hiz_yuzdesi)
    pwm_sol.ChangeDutyCycle(hiz_yuzdesi)
    ──────────────────────────────────────────────────────────────────────────
    """
    # ── DUMMY — terminale yaz ──────────────────────────────────────────────
    print(f"    [MOTOR] ▶  Güç: %{hiz_yuzdesi}  (PWM: {hiz_yuzdesi}/100)")


def direksiyonu_cevir(aci: float) -> None:
    """
    Direksiyon servosunu verilen açıya çevirir.

    Parametre:
        aci: negatif = sağa, pozitif = sola (derece)
             Örn: +45 → sola, 0 → düz, -45 → sağa

    TODO (Emir & Alara): Aşağıdaki dummy kodu gerçek servo komutuyla değiştir.
    ──────────────────────────────────────────────────────────────────────────
    # Servo için PWM duty cycle hesabı:
    #   Merkez (0°)  = 7.5 ms  →  duty ≈ 7.5  (50Hz'de 1.5ms pulse)
    #   Sol   (+45°) = 10.0 ms →  duty ≈ 10.0
    #   Sağ   (-45°) = 5.0 ms  →  duty ≈ 5.0
    duty = 7.5 + (aci / 90.0) * 5.0          # -90° → 2.5, +90° → 12.5
    duty = max(2.5, min(12.5, duty))          # Servonun güvenli aralığı
    pwm_servo.ChangeDutyCycle(duty)
    ──────────────────────────────────────────────────────────────────────────
    """
    # ── DUMMY — terminale yaz ──────────────────────────────────────────────
    yon = "SOL ◄" if aci > 0 else ("SAĞ ►" if aci < 0 else "DÜZ ↑")
    print(f"    [SERVO] {yon}  Açı: {aci:+.1f}°")


def fren_yap() -> None:
    """
    Motorları durdurur (fren / coast modu).

    TODO (Emir & Alara): Aşağıdaki dummy kodu gerçek GPIO komutuyla değiştir.
    ──────────────────────────────────────────────────────────────────────────
    # Hızlı fren (brake): IN1=IN2=HIGH → L298N kısa devre freni
    GPIO.output(PIN_IN1, GPIO.HIGH)
    GPIO.output(PIN_IN2, GPIO.HIGH)
    GPIO.output(PIN_IN3, GPIO.HIGH)
    GPIO.output(PIN_IN4, GPIO.HIGH)
    pwm_sag.ChangeDutyCycle(0)
    pwm_sol.ChangeDutyCycle(0)
    # Alternatif — motor serbest (coast): IN1=IN2=LOW
    ──────────────────────────────────────────────────────────────────────────
    """
    # ── DUMMY — terminale yaz ──────────────────────────────────────────────
    print("    [FREN]  ■  MOTORLAR DURDURULDU")


# ═══════════════════════════════════════════════════════════════════════════
#  Sapma → Servo açısı dönüştürücü
# ═══════════════════════════════════════════════════════════════════════════

def sapma_to_aci(sapma: int) -> float:
    """
    Şerit takibinden gelen piksel sapmasını servo açısına çevirir.

    Sapma:  pozitif → araç sola kaymış → sola dön (pozitif açı)
            negatif → araç sağa kaymış → sağa dön (negatif açı)

    TODO (Emir & Alara): KAZANIM sabitini kendi aracınıza göre ayarlayın.
    Küçük araçta 0.3–0.5, büyük araçta 0.1–0.2 civarı başlangıç için uygundur.
    """
    KAZANIM = 0.4   # Piksel başına derece; aşırı salınım yaparsa küçült
    aci = sapma * KAZANIM
    # Servo mekanik limitlerini aşma
    return max(SERVO_MAX_SAG, min(SERVO_MAX_SOL, aci))


# ═══════════════════════════════════════════════════════════════════════════
#  ROS2 Node Sınıfı
# ═══════════════════════════════════════════════════════════════════════════

class MotorDinleyiciNode(Node):
    """
    /arac_komut ve /serit_sapma topic'lerini dinler,
    gelen mesajlara göre motorlara ve servoya komut verir.
    """

    def __init__(self):
        super().__init__("motor_dinleyici")

        # ── /arac_komut abonesi ────────────────────────────────────────────
        # ros2_beyin_yayin.py'nin yayınladığı yüksek seviye kararları alır.
        # queue_size=10: mesaj yığılırsa en fazla 10 bekletilir, eskisi atılır.
        self.komut_sub = self.create_subscription(
            String, "/arac_komut", self._komut_callback, 10
        )

        # ── /serit_sapma abonesi ───────────────────────────────────────────
        # Şerit takibinden gelen anlık sapma değerini alır ("SAPMA:XX").
        # Bu değer sürekli yayınlanır; her mesajda servo güncellenir.
        self.sapma_sub = self.create_subscription(
            String, "/serit_sapma", self._sapma_callback, 10
        )

        # Mevcut hız durumunu hatırlayalım (sapma gelince üzerine yazmasın)
        self._son_hiz = 0

        self.get_logger().info("━━━ MotorDinleyiciNode başlatıldı ━━━")
        self.get_logger().info("  Dinleniyor: /arac_komut | /serit_sapma")

    # ── /arac_komut callback ──────────────────────────────────────────────

    def _komut_callback(self, msg: String) -> None:
        """
        Gelen komut mesajını ayrıştırır ve uygun donanım fonksiyonunu çağırır.
        ros2_beyin_yayin.py'nin yayınladığı tam mesaj listesi:
            DUR | PARK_ET | HIZ:XX | YESIL_ISIK | BEKLEME_BITTI |
            SOLLAMA | SAGA_DON | PARK_TABELASI
        """
        komut = msg.data.strip()
        self.get_logger().info(f"[KOMUT ALINDI] '{komut}'")

        # ── Dur / Fren komutları ──────────────────────────────────────────
        if komut in ("DUR", "PARK_ET"):
            # "DUR"     → Yaya geçidi veya hemzemin geçit; 5 sn duracak
            # "PARK_ET" → Kırmızı park alanı bulundu; yarışma bitti
            self.get_logger().warn(f"FREN → {komut}")
            fren_yap()
            direksiyonu_cevir(SERVO_ORTA)   # Direksiyonu düze al
            self._son_hiz = 0

        # ── Bekleme bitti / Yeşil ışık → normal hıza geç ─────────────────
        elif komut in ("YESIL_ISIK", "BEKLEME_BITTI"):
            self.get_logger().info(f"HAREKET → {komut}")
            self._son_hiz = HIZ_NORMAL
            motorlara_guc_ver(self._son_hiz)

        # ── HIZ:XX komutu — örn. "HIZ:40" veya "HIZ:20" ──────────────────
        elif komut.startswith("HIZ:"):
            try:
                hiz = int(komut.split(":")[1])
                self._son_hiz = hiz
                motorlara_guc_ver(hiz)
            except ValueError:
                self.get_logger().error(f"Geçersiz hız formatı: '{komut}'")

        # ── Sollama manevrası (Görev 5) ───────────────────────────────────
        # Sol şerite kaymak için servoya açı verilir; motor hızı biraz düşer.
        # Manevranın tamamını zaman bazlı burada ya da beyin node'unda yönet.
        elif komut == "SOLLAMA":
            self.get_logger().info("SOLLAMA → Sol şerite kayılıyor")
            direksiyonu_cevir(SERVO_MAX_SOL)    # Sola kayma başlat
            motorlara_guc_ver(35)               # Manevra hızı

        # ── Çıkmaz yol — sağa dönüş (Görev 6) ───────────────────────────
        elif komut == "SAGA_DON":
            self.get_logger().warn("ÇIKMAZ YOL → Sağa dönüş")
            fren_yap()
            direksiyonu_cevir(SERVO_MAX_SAG)    # Tam sağa
            motorlara_guc_ver(30)               # Dönüş hızı
            self._son_hiz = 30

        # ── Park tabelası görüldü — yavaşla, kırmızı alanı bekle ─────────
        elif komut == "PARK_TABELASI":
            self.get_logger().info("PARK TABELASI → Yavaşlıyor")
            self._son_hiz = 25
            motorlara_guc_ver(self._son_hiz)

        else:
            self.get_logger().warn(f"Tanınmayan komut: '{komut}' — yoksayıldı")

    # ── /serit_sapma callback ─────────────────────────────────────────────

    def _sapma_callback(self, msg: String) -> None:
        """
        Şerit takibinden gelen "SAPMA:XX" mesajını ayrıştırır.
        XX değerini servo açısına çevirip direksiyonu günceller.

        Not: Bu callback çok sık çağrılır (kare başına).
             Araç duruyorsa (hiz=0) direksiyonu hareket ettirme.
        """
        if self._son_hiz == 0:
            return   # Araç duruyorken servo sallama

        try:
            sapma = int(msg.data.split(":")[1])
        except (IndexError, ValueError):
            self.get_logger().warn(f"Geçersiz sapma formatı: '{msg.data}'")
            return

        aci = sapma_to_aci(sapma)
        direksiyonu_cevir(aci)

    # ── Node kapanırken ────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        # Güvenlik: kapanırken motorları durdur
        fren_yap()
        # TODO: GPIO.cleanup()   ← gerçek GPIO kullanılıyorsa mutlaka ekle
        self.get_logger().info("MotorDinleyiciNode kapatıldı — motorlar durduruldu.")
        super().destroy_node()


# ═══════════════════════════════════════════════════════════════════════════
#  Giriş noktası — terminalden: python3 ros2_motor_dinleyici.py
# ═══════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)

    node = MotorDinleyiciNode()
    try:
        # spin(): mesaj geldikçe callback'leri tetikler; Ctrl+C'ye kadar çalışır
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[BİTİŞ] Ctrl+C alındı, node kapatılıyor...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
