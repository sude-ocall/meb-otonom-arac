"""
baslat.py
MEB Otonom Araç — ROS2'suz Tek Dosya Başlatıcı

Beyin (kamera + YOLO + şerit takibi) ve Motor (GPIO + diferansiyel sürüş)
mantığını tek process'te birleştirir. ROS2 gerektirmez.

Kullanım:
    python3 baslat.py
"""

import sys
import os
import time
import threading
import queue
from collections import deque

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nesne_beyni import (modeli_yukle, tahmin_yap,
                          yesil_isik_var_mi, dur_komutu_var_mi,
                          cikmaz_yol_var_mi, park_tabelasi_var_mi,
                          kirmizi_park_alani_bul, park_alani_yonu,
                          YAYA_YAKIN_MIN_ALAN)
from serit_beyni import otonom_beyin
from gorev_dedektor import (hiz_tumsek_var_mi, hemzemin_var_mi,
                             turuncu_arac_var_mi)
from komutlar import (Komut,
                      HIZ_NORMAL, HIZ_YAVAS, HIZ_PARK, HIZ_PARK_SON,
                      HIZ_DONUS, HIZ_SOLLAMA)
from buton import (buton_kullanilabilir_mi, buton_hazirla,
                   buton_basildi_mi_bekle, buton_temizle)


# ═══════════════════════════════════════════════════════════════════════════
#  AYARLAR
# ═══════════════════════════════════════════════════════════════════════════
KAMERA_INDEX     = 0
FRAME_GENISLIK   = 640
FRAME_YUKSEKLIK  = 480
GORUNTU_GOSTER   = False

# Temporal kararlılık
KARARLI_PENCERE  = 5
KARARLI_ESIK     = 3

# Tabela cooldown
TABELA_COOLDOWN  = 12

# Tespit periyodu (her N karede 1 YOLO çalıştır)
TESPIT_PERIYOT   = 3

# Yarış süresi (kılavuz 4.4)
YARIS_SURESI_SN  = 240

# Görev 3: hız tümseği
TUMSEK_YAKINLIK_ESIGI  = 0.85
TUMSEK_YAVAS_SURE_SN   = 2.5

# Görev 4: hemzemin geçit
HEMZEMIN_YAKINLIK_ESIGI = 0.85

# Görev 5: sollama
SOLLAMA_TETIK_ALAN     = 6000
SOLLAMA_COOLDOWN       = 12

# Motor PWM ayarları
PWM_MAX         = 70
SAPMA_KAZANIM   = 0.4
HIZ_EWMA        = 0.6
HIZ_MIN_FARK    = 3

# Manevra süreleri
SURE_SAGA_DON        = 1.5
SURE_SOLLAMA_FAZ_A   = 1.5
SURE_SOLLAMA_FAZ_B   = 3.5
SURE_SOLLAMA_FAZ_C   = 5.0
SOLLAMA_KAYMA_ORANI  = 0.4

# Kamera pipeline (None = USB kamera)
KAMERA_GSTREAMER = None


# ═══════════════════════════════════════════════════════════════════════════
#  MOTOR KONTROL SINIFI
# ═══════════════════════════════════════════════════════════════════════════
def kistir(deger, min_deg, max_deg):
    return max(min_deg, min(max_deg, deger))


class MotorKontrol:
    """L298N + 4 motor diferansiyel sürüş kontrolü."""

    def __init__(self):
        self._cruise_hiz    = 0
        self._sol_son_pwm   = 0.0
        self._sag_son_pwm   = 0.0
        self._manevra_bitis = 0.0
        self._sollama_baslangic = 0.0
        self._gpio_hazir = False

        # GPIO kur
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            PIN_IN1, PIN_IN2, PIN_ENA = 17, 27, 18
            PIN_IN3, PIN_IN4, PIN_ENB = 22, 23, 24

            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup([PIN_IN1, PIN_IN2, PIN_ENA, PIN_IN3, PIN_IN4, PIN_ENB], GPIO.OUT)
            self._pwm_sol = GPIO.PWM(PIN_ENA, 1000)
            self._pwm_sag = GPIO.PWM(PIN_ENB, 1000)
            self._pwm_sol.start(0)
            self._pwm_sag.start(0)

            self._PIN_IN1 = PIN_IN1
            self._PIN_IN2 = PIN_IN2
            self._PIN_IN3 = PIN_IN3
            self._PIN_IN4 = PIN_IN4
            self._gpio_hazir = True
            print("[MOTOR] GPIO hazır — motorlar kontrol edilebilir")
        except (ImportError, RuntimeError) as e:
            print(f"[MOTOR] GPIO yüklenemedi ({e}) — simülasyon modu")

    def _sol_motor_set(self, pwm):
        if not self._gpio_hazir:
            return
        GPIO = self._GPIO
        pwm = max(-PWM_MAX, min(PWM_MAX, pwm))
        if pwm > 0:
            GPIO.output(self._PIN_IN1, GPIO.HIGH)
            GPIO.output(self._PIN_IN2, GPIO.LOW)
            self._pwm_sol.ChangeDutyCycle(pwm)
        elif pwm < 0:
            GPIO.output(self._PIN_IN1, GPIO.LOW)
            GPIO.output(self._PIN_IN2, GPIO.HIGH)
            self._pwm_sol.ChangeDutyCycle(abs(pwm))
        else:
            GPIO.output(self._PIN_IN1, GPIO.LOW)
            GPIO.output(self._PIN_IN2, GPIO.LOW)
            self._pwm_sol.ChangeDutyCycle(0)

    def _sag_motor_set(self, pwm):
        if not self._gpio_hazir:
            return
        GPIO = self._GPIO
        pwm = max(-PWM_MAX, min(PWM_MAX, pwm))
        if pwm > 0:
            GPIO.output(self._PIN_IN3, GPIO.HIGH)
            GPIO.output(self._PIN_IN4, GPIO.LOW)
            self._pwm_sag.ChangeDutyCycle(pwm)
        elif pwm < 0:
            GPIO.output(self._PIN_IN3, GPIO.LOW)
            GPIO.output(self._PIN_IN4, GPIO.HIGH)
            self._pwm_sag.ChangeDutyCycle(abs(pwm))
        else:
            GPIO.output(self._PIN_IN3, GPIO.LOW)
            GPIO.output(self._PIN_IN4, GPIO.LOW)
            self._pwm_sag.ChangeDutyCycle(0)

    def fren_yap(self):
        if self._gpio_hazir:
            GPIO = self._GPIO
            GPIO.output(self._PIN_IN1, GPIO.HIGH)
            GPIO.output(self._PIN_IN2, GPIO.HIGH)
            GPIO.output(self._PIN_IN3, GPIO.HIGH)
            GPIO.output(self._PIN_IN4, GPIO.HIGH)
            self._pwm_sol.ChangeDutyCycle(0)
            self._pwm_sag.ChangeDutyCycle(0)
        self._sol_son_pwm = 0.0
        self._sag_son_pwm = 0.0
        self._sollama_baslangic = 0.0
        print("[MOTOR] FREN")

    def diff_uygula(self, sol_hedef, sag_hedef):
        sol_hedef = kistir(sol_hedef, -PWM_MAX, PWM_MAX)
        sag_hedef = kistir(sag_hedef, -PWM_MAX, PWM_MAX)
        sol_yeni = HIZ_EWMA * self._sol_son_pwm + (1 - HIZ_EWMA) * sol_hedef
        sag_yeni = HIZ_EWMA * self._sag_son_pwm + (1 - HIZ_EWMA) * sag_hedef
        if (abs(sol_yeni - self._sol_son_pwm) >= HIZ_MIN_FARK or
            abs(sag_yeni - self._sag_son_pwm) >= HIZ_MIN_FARK):
            self._sol_son_pwm = sol_yeni
            self._sag_son_pwm = sag_yeni
            self._sol_motor_set(sol_yeni)
            self._sag_motor_set(sag_yeni)

    def komut_isle(self, komut):
        """Yüksek seviye komutu işle."""
        print(f"[MOTOR] Komut: {komut}")

        if komut in (Komut.DUR, Komut.PARK_ET):
            self._cruise_hiz = 0
            self.fren_yap()

        elif komut in (Komut.YESIL_ISIK, Komut.BEKLEME_BITTI):
            self._cruise_hiz = HIZ_NORMAL
            self.diff_uygula(HIZ_NORMAL, HIZ_NORMAL)

        elif komut.startswith("HIZ:"):
            try:
                self._cruise_hiz = int(komut.split(":")[1])
            except (ValueError, IndexError):
                pass

        elif komut == Komut.SOLLAMA:
            if self._sollama_baslangic > 0:
                return
            print("[MOTOR] SOLLAMA başladı (3 fazlı)")
            self._cruise_hiz = HIZ_SOLLAMA
            self._sollama_baslangic = time.time()
            self._manevra_bitis = self._sollama_baslangic + SURE_SOLLAMA_FAZ_C

        elif komut == Komut.SOLLAMA_BITTI:
            if self._sollama_baslangic == 0.0:
                return
            gecen = time.time() - self._sollama_baslangic
            if gecen < SURE_SOLLAMA_FAZ_A:
                return
            if gecen >= SURE_SOLLAMA_FAZ_B:
                return
            atlanan = SURE_SOLLAMA_FAZ_B - gecen
            self._sollama_baslangic -= atlanan
            self._manevra_bitis = self._sollama_baslangic + SURE_SOLLAMA_FAZ_C

        elif komut == Komut.SAGA_DON:
            self._cruise_hiz = HIZ_DONUS
            self.diff_uygula(+HIZ_DONUS, -HIZ_DONUS)
            self._manevra_bitis = time.time() + SURE_SAGA_DON

        elif komut == Komut.PARK_TABELASI:
            self._cruise_hiz = HIZ_PARK

    def sapma_isle(self, sapma):
        """Şerit sapmasını diferansiyel hıza çevir."""
        if self._cruise_hiz == 0:
            return
        if self._sollama_baslangic > 0:
            return
        if time.time() < self._manevra_bitis:
            return
        delta = SAPMA_KAZANIM * sapma
        self.diff_uygula(self._cruise_hiz - delta, self._cruise_hiz + delta)

    def sollama_tik(self):
        """Sollama fazlarını yürüt — ana döngüden çağrılır."""
        if self._sollama_baslangic == 0.0:
            return
        gecen = time.time() - self._sollama_baslangic
        if gecen < SURE_SOLLAMA_FAZ_A:
            self.diff_uygula(HIZ_SOLLAMA * SOLLAMA_KAYMA_ORANI, HIZ_SOLLAMA)
        elif gecen < SURE_SOLLAMA_FAZ_B:
            self.diff_uygula(HIZ_SOLLAMA, HIZ_SOLLAMA)
        elif gecen < SURE_SOLLAMA_FAZ_C:
            self.diff_uygula(HIZ_SOLLAMA, HIZ_SOLLAMA * SOLLAMA_KAYMA_ORANI)
        else:
            print(f"[MOTOR] Sollama tamamlandı ({gecen:.1f}sn)")
            self._sollama_baslangic = 0.0
            self._cruise_hiz = HIZ_NORMAL

    def temizle(self):
        self.fren_yap()
        if self._gpio_hazir:
            self._GPIO.cleanup()
        print("[MOTOR] GPIO temizlendi")


# ═══════════════════════════════════════════════════════════════════════════
#  BEYİN SINIFI
# ═══════════════════════════════════════════════════════════════════════════
class AracBeyni:
    """Kamera → YOLO → Karar → Motor."""

    ISIK_BEKLE = "ISIK_BEKLE"
    NORMAL     = "NORMAL"
    DUR_BEKLE  = "DUR_BEKLE"
    PARK_ARAMA = "PARK_ARAMA"

    def __init__(self, motor: MotorKontrol):
        self.motor = motor

        # Kamera
        if KAMERA_GSTREAMER:
            self.cap = cv2.VideoCapture(KAMERA_GSTREAMER, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(KAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FOURCC,
                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_GENISLIK)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_YUKSEKLIK)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Kamera açılamadı!")
        print("[BEYİN] Kamera açıldı")

        # YOLO modeli
        modeli_yukle()
        print("[BEYİN] Model yüklendi")

        # Durum
        self.durum             = self.ISIK_BEKLE
        self.son_tabela_zamani = 0.0
        self.bekleme_bitis     = 0.0
        self.yaris_baslangic   = 0.0
        self.yaris_bitti       = False
        self._tumsek_donus_zamani = 0.0
        self._son_sollama_zamani  = 0.0
        self._sollama_aktif       = False
        self._sollama_bitti_yayinlandi = False
        self._park_son_hiza_dusuruldu = False
        self._tabela_gecmis = deque(maxlen=KARARLI_PENCERE)
        self._kare_sayisi   = 0
        self._son_tespitler = []
        self._son_dur_tekrar = 0.0

        # Kamera thread
        self._frame_kuyrugu = queue.Queue(maxsize=1)
        self._kapaniyor = threading.Event()
        self._kamera_thread = threading.Thread(target=self._kamera_dongusu, daemon=True)
        self._kamera_thread.start()

        print("[BEYİN] Hazır — yeşil ışık bekleniyor...")

    def _t(self):
        if self.yaris_baslangic <= 0:
            return "[T-pre] "
        return f"[T+{time.time() - self.yaris_baslangic:5.1f}s] "

    def _kararli(self, sinif):
        return sum(sinif in s for s in self._tabela_gecmis) >= KARARLI_ESIK

    def _kamera_dongusu(self):
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

    def _komut(self, komut):
        """Komutu motora gönder."""
        print(f"{self._t()}[KOMUT] {komut}")
        self.motor.komut_isle(komut)

    def _sapma(self, sapma_degeri):
        """Sapma değerini motora gönder."""
        self.motor.sapma_isle(sapma_degeri)

    def kare_isle(self):
        """Ana döngüden her iterasyonda çağrılır."""
        # Sollama fazlarını yürüt
        self.motor.sollama_tik()

        try:
            frame = self._frame_kuyrugu.get_nowait()
        except queue.Empty:
            return

        su_an = time.time()

        # Yarış sonu kontrolü
        if (self.yaris_baslangic > 0 and not self.yaris_bitti and
                (su_an - self.yaris_baslangic) >= YARIS_SURESI_SN):
            print(f"{self._t()}4 DAKİKA DOLDU — yarış sonlandırılıyor")
            self._komut(Komut.DUR)
            self.yaris_bitti = True
            self._son_dur_tekrar = su_an

        if self.yaris_bitti:
            if su_an - self._son_dur_tekrar >= 1.0:
                self._komut(Komut.DUR)
                self._son_dur_tekrar = su_an
            return

        # Görev 3: tümsek süresi dolduysa normale dön
        if self._tumsek_donus_zamani > 0 and su_an >= self._tumsek_donus_zamani:
            print("Hız tümseği geçildi — normal hıza dönülüyor")
            self._komut(Komut.hiz(HIZ_NORMAL))
            self._tumsek_donus_zamani = 0.0

        # DUR_BEKLE
        if self.durum == self.DUR_BEKLE:
            if su_an < self.bekleme_bitis:
                return
            self._komut(Komut.BEKLEME_BITTI)
            self._komut(Komut.hiz(HIZ_NORMAL))
            self.durum = self.NORMAL

        # Frame skip
        self._kare_sayisi += 1
        tespit_yap = (self._kare_sayisi % TESPIT_PERIYOT == 0)

        # YOLO tespiti
        if tespit_yap:
            tespitler = tahmin_yap(frame)
            self._son_tespitler = tespitler
            siniflar = {ad for ad, _, _ in tespitler}
            self._tabela_gecmis.append(siniflar)
        else:
            tespitler = self._son_tespitler
            siniflar = {ad for ad, _, _ in tespitler}

        # ISIK_BEKLE
        if self.durum == self.ISIK_BEKLE:
            if yesil_isik_var_mi(tespitler, frame):
                self.yaris_baslangic = su_an
                print(f"{self._t()}YEŞİL IŞIK — yarışma başlıyor!")
                self._komut(Komut.YESIL_ISIK)
                self._komut(Komut.hiz(HIZ_NORMAL))
                self.durum = self.NORMAL
            return

        # PARK_ARAMA
        if self.durum == self.PARK_ARAMA:
            var, merkez, alan, icinde_mi = kirmizi_park_alani_bul(frame)
            if var:
                yon = park_alani_yonu(frame, merkez)
                if icinde_mi:
                    bitirme = su_an - self.yaris_baslangic
                    print(f"{self._t()}PARK TAMAMLANDI — süre={bitirme:.1f}s")
                    self._komut(Komut.PARK_ET)
                    return
                _, cy = merkez
                if cy > frame.shape[0] * 0.65 and not self._park_son_hiza_dusuruldu:
                    self._komut(Komut.hiz(HIZ_PARK_SON))
                    self._park_son_hiza_dusuruldu = True
                self._sapma(yon)
                return

        # NORMAL: tabela mantığı
        cooldown_ok = (su_an - self.son_tabela_zamani) > TABELA_COOLDOWN

        if cooldown_ok and siniflar:
            if (dur_komutu_var_mi(tespitler, frame, min_alan=YAYA_YAKIN_MIN_ALAN)
                    and any(self._kararli(s) for s in ("YayaGecidi", "dur", "IsikTabelasi"))):
                print(f"{self._t()}YAYA GEÇİDİ — 5 sn dur")
                self._komut(Komut.DUR)
                self.durum = self.DUR_BEKLE
                self.bekleme_bitis = su_an + 5
                self.son_tabela_zamani = su_an
                return

            elif cikmaz_yol_var_mi(tespitler) and self._kararli("girilmez"):
                print(f"{self._t()}GİRİLMEZ — sağa dönüş")
                self._komut(Komut.SAGA_DON)
                self.son_tabela_zamani = su_an

            elif park_tabelasi_var_mi(tespitler) and self._kararli("park"):
                print(f"{self._t()}PARK TABELASI — kırmızı alan aranıyor")
                self._komut(Komut.PARK_TABELASI)
                self.durum = self.PARK_ARAMA
                self.son_tabela_zamani = su_an

        # HSV görevler
        if self.durum == self.NORMAL and tespit_yap:
            if cooldown_ok:
                var_hz, yakin_hz = hemzemin_var_mi(frame)
                if var_hz and yakin_hz >= HEMZEMIN_YAKINLIK_ESIGI:
                    print(f"{self._t()}HEMZEMİN GEÇİT — 5 sn dur")
                    self._komut(Komut.DUR)
                    self.durum = self.DUR_BEKLE
                    self.bekleme_bitis = su_an + 5
                    self.son_tabela_zamani = su_an
                    return

            if self._tumsek_donus_zamani == 0.0:
                var_t, yakin_t = hiz_tumsek_var_mi(frame)
                if var_t and yakin_t >= TUMSEK_YAKINLIK_ESIGI:
                    print(f"{self._t()}HIZ TÜMSEĞİ — yavaşla")
                    self._komut(Komut.hiz(HIZ_YAVAS))
                    self._tumsek_donus_zamani = su_an + TUMSEK_YAVAS_SURE_SN

            var_o, alan_o, _ = turuncu_arac_var_mi(frame)

            if self._sollama_aktif and not self._sollama_bitti_yayinlandi:
                gecen = su_an - self._son_sollama_zamani
                if gecen > 1.0 and not var_o:
                    self._komut(Komut.SOLLAMA_BITTI)
                    self._sollama_bitti_yayinlandi = True

            sollama_cooldown_ok = (su_an - self._son_sollama_zamani) > SOLLAMA_COOLDOWN
            if sollama_cooldown_ok and not self._sollama_aktif:
                if var_o and alan_o >= SOLLAMA_TETIK_ALAN:
                    print(f"TURUNCU ARAÇ (alan={alan_o}) — SOLLAMA")
                    self._komut(Komut.SOLLAMA)
                    self._son_sollama_zamani = su_an
                    self._sollama_aktif = True
                    self._sollama_bitti_yayinlandi = False

            if (self._sollama_aktif and
                    (su_an - self._son_sollama_zamani) >= SOLLAMA_COOLDOWN):
                self._sollama_aktif = False
                self._sollama_bitti_yayinlandi = False

        # Şerit takibi
        try:
            _, sapma = otonom_beyin(frame)
            self._sapma(sapma)
        except Exception as e:
            pass

    def kapat(self):
        self._kapaniyor.set()
        self._kamera_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
        self.motor.temizle()
        print("[BEYİN] Kapatıldı")


# ═══════════════════════════════════════════════════════════════════════════
#  ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("════════════════════════════════════════════════════════════")
    print("  MEB OTONOM ARAÇ — TEK DOSYA BAŞLATICI (ROS2'suz)")
    print("════════════════════════════════════════════════════════════")

    # Görev 1: Buton ile başlatma (+50 puan)
    if buton_kullanilabilir_mi() and buton_hazirla():
        print("[GÖREV 1] Buton modu — butona basılması bekleniyor...")
        buton_basildi_mi_bekle()
    else:
        print("[GÖREV 1] Buton yok — doğrudan başlatılıyor")

    motor = MotorKontrol()

    try:
        beyin = AracBeyni(motor)
    except RuntimeError as e:
        print(f"[HATA] Başlatılamadı: {e}")
        motor.temizle()
        buton_temizle()
        sys.exit(1)

    print("[BAŞLADI] Ana döngü çalışıyor...")

    try:
        while True:
            beyin.kare_isle()
            time.sleep(0.01)  # ~100 Hz döngü
    except KeyboardInterrupt:
        print("\n[BİTİŞ] Ctrl+C — kapatılıyor...")
    finally:
        beyin.kapat()
        buton_temizle()

    print("════════════════════════════════════════════════════════════")
    print("  YARIŞMA SONLANDI")
    print("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
