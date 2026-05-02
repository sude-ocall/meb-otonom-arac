"""
main.py
Ana yarışma döngüsü.
Her frame'de:
  1. YOLOv8 tespiti  (nesne_beyni)
  2. Görev mantığı   (tabela/ışık → motor komutu)
  3. Şerit takibi    (serit_beyni → direksiyon açısı)
"""
import cv2
import time
from serit_beyni  import otonom_beyin
from nesne_beyni  import (modeli_yukle, tahmin_yap,
                           yesil_isik_var_mi, dur_komutu_var_mi,
                           hiz_tumseği_var_mi, sollama_serbest_var_mi,
                           cikmaz_yol_var_mi, park_tabelasi_var_mi,
                           kirmizi_park_alani_var_mi, gorsele_ciz)
from motor_surucu import motorlari_hazirla, ileri_git, dur, direksiyon_cevir

# ── Sabitler (kılavuz 4.4) ─────────────────────────────────────────────────
YARIS_SURESI     = 240   # Azami tur süresi: 4 dakika
NORMAL_HIZ       = 40    # Normal ilerleme hızı
YAVAS_HIZ        = 20    # Hız tümseği geçiş hızı
TABELA_COOLDOWN  = 15    # Bir tabelaya tepki verdikten sonra beklenecek süre (sn)
#   → Aynı tabelayı art arda tekrar algılayıp sürekli durmasın diye.


def _kamera_dreni(cap, sure: float) -> None:
    """
    sure saniye boyunca kamera karesini oku ama işleme.
    5 saniyelik bekleme sırasında buffer dolmasın diye çağrılır.
    """
    bitis = time.time() + sure
    while time.time() < bitis:
        cap.read()


def baslat():
    motorlari_hazirla()
    modeli_yukle("best.pt")

    cap = cv2.VideoCapture(0)   # Raspberry Pi kamerası için 0; harici için 1
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı.")
        return

    yarisma_basladi  = False
    yaris_baslangic  = None
    son_tabela_zamani = 0       # Cooldown takibi

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (400, 300))

        # ── YOLOv8 tespiti (her frame) ─────────────────────────────────────
        tespitler = tahmin_yap(frame)

        # ── GÖREV 1: Trafik ışığı — yeşil görene kadar bekle ───────────────
        if not yarisma_basladi:
            if yesil_isik_var_mi(tespitler, frame):
                print("[BAŞLANGIÇ] Yeşil ışık! Hareket başlıyor.")
                yarisma_basladi = True
                yaris_baslangic = time.time()
            else:
                dur()
                cv2.imshow("Gorus", gorsele_ciz(frame, tespitler))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue     # Aşağıdaki görev koduna geçme

        su_an = time.time()

        # Azami süre kontrolü — kılavuz 4.4
        if (su_an - yaris_baslangic) >= YARIS_SURESI:
            dur()
            print("[SÜRE] 4 dakika doldu, tur sonlandırılıyor.")
            break

        # ── GÖREV 7: Kırmızı park alanı — park et ve bitir ────────────────
        if kirmizi_park_alani_var_mi(tespitler):
            dur()
            print("[BİTİŞ] Kırmızı park alanı bulundu. Yarışma tamamlandı.")
            break

        # ── Tabela görev mantığı (cooldown korumalı) ───────────────────────
        cooldown_bitti = (su_an - son_tabela_zamani) > TABELA_COOLDOWN

        if cooldown_bitti:

            # GÖREV 2 & 4: Yaya geçidi veya hemzemin geçit — 5 sn dur
            if dur_komutu_var_mi(tespitler):
                dur()
                print("[GÖREV 2/4] Dur tabelası — 5 sn bekleniyor.")
                _kamera_dreni(cap, 5)       # Kamera buffer'ı dolu kalmasın
                son_tabela_zamani = su_an   # Cooldown'u sıfırla
                continue                    # Frame'i yeniden işle; şerit kodu atla

            # GÖREV 3: Hız tümseği — yavaşla, geç, hızlan
            elif hiz_tumseği_var_mi(tespitler):
                print("[GÖREV 3] Hız tümseği — yavaş geçiliyor.")
                ileri_git(YAVAS_HIZ)
                son_tabela_zamani = su_an

            # GÖREV 5: Sollama serbest — sol şerit → sollamayı tamamla → geri dön
            elif sollama_serbest_var_mi(tespitler):
                print("[GÖREV 5] Sollama serbest — manevra başlıyor.")
                direksiyon_cevir(80)        # Sola kayma
                ileri_git(NORMAL_HIZ)
                _kamera_dreni(cap, 1.2)
                direksiyon_cevir(0)         # Düz — araç sollanıyor
                ileri_git(NORMAL_HIZ)
                _kamera_dreni(cap, 1.5)
                direksiyon_cevir(-80)       # Sağa dön — kendi şeridine geri
                ileri_git(NORMAL_HIZ)
                _kamera_dreni(cap, 1.2)
                son_tabela_zamani = su_an

            # GÖREV 6: Çıkmaz yol — sağa dön, yola devam et
            elif cikmaz_yol_var_mi(tespitler):
                dur()
                print("[GÖREV 6] Çıkmaz yol — sağa dönülüyor.")
                direksiyon_cevir(-100)
                ileri_git(30)
                _kamera_dreni(cap, 1.5)
                son_tabela_zamani = su_an

            # GÖREV 7 öncesi park tabelası uyarısı
            elif park_tabelasi_var_mi(tespitler):
                print("[GÖREV 7] Park tabelası görüldü, kırmızı alan aranıyor.")
                son_tabela_zamani = su_an

        # ── Şerit takibi — tabela yoksa veya cooldown içindeyse ───────────
        res, sapma = otonom_beyin(frame)
        if res is not None:
            ileri_git(NORMAL_HIZ)
            direksiyon_cevir(sapma)
            cv2.imshow("Gorus", gorsele_ciz(res, tespitler))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dur()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    baslat()
