"""
main.py
Ana yarışma döngüsü.

Modeldeki sınıf eksikliği nedeniyle aktif görevler:
  - Görev 1 — Trafik ışığı (yesil/kirmizi/sari sınıfları model'de var)
  - Görev 2 — Yaya geçidi (YayaGecidi sınıfı model'de var)
  - Görev 6 — Çıkmaz yol (girilmez tabelası ile, modelde 'CikmazYol' yok)
  - Görev 7 — Park (mavi 'park' tabelası + HSV ile yerdeki kırmızı alan)
  - Görev 8 — Bölge tamamlama (şerit takibiyle otomatik)

Modelde olmadığı için DEVRE DIŞI görevler:
  - Görev 3 (HizTumseği), 4 (HemzeminGecit), 5 (SollamaSerbest)

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
                           cikmaz_yol_var_mi, park_tabelasi_var_mi,
                           kirmizi_park_alani_bul, park_alani_yonu,
                           gorsele_ciz)
from motor_surucu import motorlari_hazirla, ileri_git, dur, direksiyon_cevir

# ── Sabitler (kılavuz 4.4) ─────────────────────────────────────────────────
YARIS_SURESI     = 240   # Azami tur süresi: 4 dakika
NORMAL_HIZ       = 40    # Normal ilerleme hızı
PARK_ARAMA_HIZ   = 25    # Park modunda yavaş ilerleme
TABELA_COOLDOWN  = 15    # Bir tabelaya tepki verdikten sonra beklenecek süre (sn)

# Park alanı bulma için minimum alan oranı — yaklaştıkça bu artar
PARK_VARDIM_ALAN_ORANI = 0.18   # Frame'in %18'i kırmızıysa "park alanına vardık"


def _kamera_dreni(cap, sure: float) -> None:
    """sure saniye boyunca kamera karesini oku ama işleme."""
    bitis = time.time() + sure
    while time.time() < bitis:
        cap.read()


def baslat():
    motorlari_hazirla()
    modeli_yukle()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı.")
        return

    yarisma_basladi  = False
    yaris_baslangic  = None
    son_tabela_zamani = 0
    park_modu        = False     # park tabelası görüldü → kırmızı arama aktif

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (400, 300))

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
                continue

        su_an = time.time()

        if (su_an - yaris_baslangic) >= YARIS_SURESI:
            dur()
            print("[SÜRE] 4 dakika doldu, tur sonlandırılıyor.")
            break

        # ── GÖREV 7: Park modu aktifse — yerdeki kırmızı alanı ara ─────────
        # Park tabelası görüldükten sonra HSV ile zemindeki üç renkten
        # SADECE kırmızı park alanını seçer; mavi/yeşil alanları yok sayar.
        if park_modu:
            var, merkez, alan = kirmizi_park_alani_bul(frame)
            if var:
                # Alan oranını ölç — yeterince yakınsak dur
                alan_orani = alan / (frame.shape[0] * frame.shape[1])
                yon = park_alani_yonu(frame, merkez)

                if alan_orani >= PARK_VARDIM_ALAN_ORANI:
                    dur()
                    print(f"[BİTİŞ] Kırmızı park alanına park edildi "
                          f"(alan oranı={alan_orani:.2f}). Yarışma tamamlandı.")
                    break

                # Henüz yakın değiliz — kırmızıya doğru yönlen
                print(f"[GÖREV 7] Kırmızı park alanı görüldü "
                      f"(yön={yon:+d}, alan={alan_orani:.2f}). Yaklaşılıyor.")
                direksiyon_cevir(yon)
                ileri_git(PARK_ARAMA_HIZ)

                cv2.imshow("Gorus", gorsele_ciz(frame, tespitler))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            # Henüz kırmızı görünmedi — yavaş yavaş ilerlemeye devam (şerit takibi)

        # ── Tabela görev mantığı (cooldown korumalı) ───────────────────────
        cooldown_bitti = (su_an - son_tabela_zamani) > TABELA_COOLDOWN

        if cooldown_bitti:

            # GÖREV 2: Yaya geçidi / dur tabelası — 5 sn dur
            if dur_komutu_var_mi(tespitler, frame):
                dur()
                print("[GÖREV 2] Dur/yaya tabelası — 5 sn bekleniyor.")
                _kamera_dreni(cap, 5)
                son_tabela_zamani = su_an
                continue

            # GÖREV 6: Çıkmaz yol (girilmez tabelası) — sağa dön
            elif cikmaz_yol_var_mi(tespitler):
                dur()
                print("[GÖREV 6] Girilmez tabelası — sağa dönülüyor.")
                direksiyon_cevir(-100)
                ileri_git(30)
                _kamera_dreni(cap, 1.5)
                son_tabela_zamani = su_an

            # GÖREV 7 öncesi: park tabelası — park modunu aç
            elif park_tabelasi_var_mi(tespitler):
                print("[GÖREV 7] Park tabelası görüldü, kırmızı alan aranıyor.")
                park_modu = True
                son_tabela_zamani = su_an

        # ── Şerit takibi — tabela yoksa veya cooldown içindeyse ───────────
        res, sapma = otonom_beyin(frame)
        if res is not None:
            hiz = PARK_ARAMA_HIZ if park_modu else NORMAL_HIZ
            ileri_git(hiz)
            direksiyon_cevir(sapma)
            cv2.imshow("Gorus", gorsele_ciz(res, tespitler))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dur()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    baslat()
