# MEB Robot Yarışması — Otonom Araç

18\. Uluslararası MEB Robot Yarışması, **Otonom Araç** kategorisi (2026) için geliştirilmiş proje.
Araç yalnızca kamera ile şerit takibi yapar; YOLOv8 + HSV ile tabelaları, trafik ışıklarını ve görev objelerini tanıyarak 8 görevi tamamlar.

**Donanım:** Raspberry Pi 4 + L298N + 4× DC gearmotor (diferansiyel sürüş, direksiyon yok) + Pi/USB kamera + GPIO 26 başlatma butonu.

**Mimari:** Tek path — `ros2_beyin_yayin.py` (algı/karar) ↔ `ros2_motor_dinleyici.py` (motor sürüş) arasında ROS2 topic'leri. Heartbeat-watchdog ile beyin koparsa motor 1 sn'de fren yapar.

---

## Dosyalar

| Dosya | Görevi |
|---|---|
| `ros2_beyin_yayin.py` | **Beyin node**. Kamera → YOLO + HSV → karar → ROS2 yayın. 240 sn süre takipçisi, durum makinesi (ISIK_BEKLE → NORMAL → DUR_BEKLE → PARK_ARAMA). |
| `ros2_motor_dinleyici.py` | **Motor node**. Komutları dinler, sol/sağ PWM diferansiyeli uygular. Watchdog (1sn timeout) + 3 fazlı sollama tick'i. |
| `nesne_beyni.py` | YOLOv8 inference + HSV doğrulama. NCNN > engine > pt sırasıyla otomatik model seçer. |
| `gorev_dedektor.py` | Modelde sınıfı olmayan görevler için HSV tespiti: hız tümseği (sarı şeritler), hemzemin geçit (beyaz ızgara), turuncu sollama aracı. |
| `serit_beyni.py` | Sliding-window şerit takibi (kuş bakışı warp + N-bant peak). Sapma değeri döner. |
| `goruntu_islem.py` | CLAHE parlama azaltma, ROI kırpma, kırmızı/mavi/sarı HSV doğrulama yardımcıları. |
| `komutlar.py` | ROS2 string komutları + topic adları + hız profilleri (tek kaynak). |
| `buton.py` | GPIO 26 buton ile yarışma başlatma (kılavuz 3.4.1, +50 puan). |
| `motor_test.py` | 4 motorun bağımsız döndüğünü test eder (Pi'da, GPIO ile). |
| `test_smoke.py` | Donanımsız smoke testler — HSV tespitleri, modül syntax, sollama faz tutarlılığı. |
| `test_kamera.py` | MacBook'ta tüm akışı simüle eder; motor yayını yoktur. |
| `test_tespit.py` | Canlı YOLO tespit tanı aracı (`+`/`-` ile eşik). |
| `yarisma_baslat.sh` | Yarışma günü başlatma — WiFi/BT kapatır, venv aktive eder, ROS2 node'larını başlatır. |

---

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pi 4: NCNN'e dönüştür (best.pt → 5-8× hızlanma)
yolo export model=best.pt format=ncnn imgsz=320
```

ROS2 (Pi 4 / Ubuntu 22.04):
```bash
sudo apt install ros-humble-rclpy ros-humble-std-msgs python3-rpi.gpio
source /opt/ros/humble/setup.bash
```

---

## Çalıştırma

### Yarışma günü (Pi 4)
```bash
sudo ./yarisma_baslat.sh
```
WiFi/BT kapatır, venv açar, motor node arka planda + beyin node ön planda başlar.
Buton donanımı varsa beyin önce butona basılmasını bekler (Görev 1, +50 puan).

### Geliştirme / kalibrasyon (MacBook)
```bash
python test_smoke.py        # donanımsız regresyon paketi
python test_tespit.py       # canlı YOLO tanı
python test_kamera.py       # tam akış simülasyonu (motor yayını yok)
```

---

## Görev → Mantık → Puan Eşlemesi

| Görev | Tetik | Davranış | Puan |
|---|---|---|---|
| 1a | Buton (GPIO 26) | Brain'den önce sürecek butonu bekler | **+50** |
| 1b | YOLO `yesil` + HSV alt-yarı doğrulama | İlk denemede hareket → 50p, ikincide 25p | **50/25** |
| 2 | YOLO `YayaGecidi` + bbox alanı `≥ YAYA_YAKIN_MIN_ALAN` | DUR + 5 sn DUR_BEKLE → BEKLEME_BITTI + HIZ:40 | **50** |
| 3 | HSV: `≥4` dikey sarı şerit + `yakınlık ≥ 0.85` | HIZ:20, 2.5 sn sonra HIZ:40 | **50** |
| 4 | HSV: `≥8` beyaz blok ızgara + `yakınlık ≥ 0.85` | DUR + 5 sn DUR_BEKLE | **50** |
| 5 | HSV turuncu blob, alan `≥ SOLLAMA_TETIK_ALAN` | 3 fazlı SOLLAMA: kay sola → düz git → kay sağa. Erken çıkış: turuncu kaybolursa SOLLAMA_BITTI | **100** |
| 6 | YOLO `girilmez` + temporal kararlılık | SAGA_DON (motor 1.5 sn tank dönüşü) | **100** |
| 7 | YOLO `park` → PARK_ARAMA durumu → HSV kırmızı zemin tespit + cy alt %35'te HIZ:15 → kırmızı dibe ulaşınca PARK_ET | Aşamalı yaklaşma + park | **100** |
| 8 | Şerit takibi (sliding-window) — `serit_beyni.otonom_beyin` | Sürekli sapma yayını | **50/bölge** |
| Bonus | `(4×60) - bitirme_süresi` | PARK_ET anında loglanır | süre |
| Diskalifiye | WiFi/BT açık → kılavuz 2.3 | `yarisma_baslat.sh` rfkill ile kapatır | — |

**Toplam üst sınır:** ~550 + bitirme süresi katsayısı.

---

## Kalibrasyon Düğmeleri (sahaya göre ayarlanacak)

### `nesne_beyni.py`
| Sabit | Varsayılan | Ne zaman değiştir |
|---|---|---|
| `GUVEN_ESIGI` | 0.35 | Yanlış tespit çok → 0.45'e çek; tespit yok → 0.25'e düşür |
| `YAYA_YAKIN_MIN_ALAN` | 4500 | Çok erken duruyor → 6000; geç duruyor → 3000 |
| `MIN_KUTU_ALANI` | 500 | Küçük tabelalar kaçıyorsa düşür |
| `SINIF_ESIGI` | per-sınıf | Çiftler karışıyorsa (sola↔saga) yükselt |

### `gorev_dedektor.py`
| Sabit | Varsayılan | Ne için |
|---|---|---|
| `SARI_ALT/UST` | H 22-35 | Hız tümseği şeritleri (saha aydınlatmasına göre kaydır) |
| `TURUNCU_ALT/UST` | H 8-20 | Sollanacak araç (sarı tümsek karışıyorsa daralt) |
| `BEYAZ_ALT/UST` | V≥200 | Hemzemin ızgara — pist beyaz değilse eşiği düşür |

### `ros2_beyin_yayin.py`
| Sabit | Varsayılan | Etkisi |
|---|---|---|
| `YARIS_SURESI_SN` | 240 | Kılavuz 4.4'e bağlı, değiştirme |
| `TABELA_COOLDOWN` | 12 | Aynı tabela tekrar tetiklenmesin |
| `TUMSEK_YAKINLIK_ESIGI` | 0.85 | Tümsek üzerine basmadan ne zaman yavaşlasın |
| `TUMSEK_YAVAS_SURE_SN` | 2.5 | Tümsek üstü süresi |
| `HEMZEMIN_YAKINLIK_ESIGI` | 0.85 | Hemzeminden 30 cm önce dur |
| `SOLLAMA_TETIK_ALAN` | 6000 | Turuncuya kaç cm kala sollama başlasın |
| `SOLLAMA_COOLDOWN` | 12 | **Manevra süresinden büyük olmalı** (smoke test bunu yakalar) |

### `ros2_motor_dinleyici.py`
| Sabit | Varsayılan | Etkisi |
|---|---|---|
| `PWM_MAX` | 70 | Motor güvenlik tavanı (9V/L298N için %100 yakar) |
| `SAPMA_KAZANIM` | 0.4 | Şerit tepki kuvveti (salınım var → 0.3, yetmiyor → 0.5) |
| `HIZ_EWMA` | 0.6 | Hız yumuşatma (titrek motor → yükselt) |
| `SURE_SOLLAMA_FAZ_A/B/C` | 1.5 / 3.5 / 5.0 sn | 3-fazlı sollama zamanlaması |
| `SOLLAMA_KAYMA_ORANI` | 0.4 | Yan kayma şiddeti (0=tek motor, 1=eşit) |
| `SURE_SAGA_DON` | 1.5 sn | Çıkmaz yol tank dönüşü |
| `KALP_TIMEOUT_SN` | 1.0 | Watchdog tetik süresi |

### `serit_beyni.py`
| Sabit | Varsayılan | Etkisi |
|---|---|---|
| `BANT_SAYISI` | 9 | Daha çok bant: pürüzsüz; gürültü çoksa düşür |
| `PEAK_MIN_PIKSEL` | 30 | Bant başına şerit kabul eşiği |
| `ARAMA_PENCERE_GEN` | 80 | Bantlar arası şerit takip yarıçapı |
| `L_ESIK_ALT` | 170 | Beyaz şerit alt sınırı (loş ışık → 140) |

---

## Mimari ve Topic'ler

```
                    ┌──────────────────────────────────┐
                    │      ros2_beyin_yayin.py         │
                    │  (kamera → YOLO/HSV → karar)     │
                    └──────────────────────────────────┘
                            │      │      │
                /arac_komut │      │      │ /serit_sapma
                 (RELIABLE) │      │      │ (BestEffort, depth=1)
                            │      │      │
                  /beyin_kalp ◄────┘      │
                  (5 Hz heartbeat)        │
                            │      │      │
                            ▼      ▼      ▼
                    ┌──────────────────────────────────┐
                    │   ros2_motor_dinleyici.py        │
                    │  (komut → diferansiyel PWM)      │
                    │  Watchdog: 1 sn'de fren          │
                    │  Sollama tick'i: 10Hz            │
                    └──────────────────────────────────┘
```

**Komut listesi** (`komutlar.py`'den):
- `DUR`, `PARK_ET` → fren
- `YESIL_ISIK`, `BEKLEME_BITTI` → cruise = HIZ_NORMAL
- `HIZ:XX` → cruise hızı güncelle
- `SOLLAMA` → 3 fazlı manevra başlat
- `SOLLAMA_BITTI` → Faz B'den C'ye atla (turuncu kaybolduğunda)
- `SAGA_DON` → 1.5 sn tank dönüşü
- `PARK_TABELASI` → cruise = HIZ_PARK
- `KALP` → heartbeat (sensor QoS, watchdog reset)

---

## Notlar

- **Kamera açısı:** Hem şeridi hem tabelaları görmeli. Şerit zeminde, tabela ~30cm yüksekte. Çok aşağı bakarsa tabela kaçar; çok yukarı bakarsa şerit görmez.
- **GPIO grup:** `sudo usermod -aG gpio,video $USER` + tekrar giriş. Yarisma_baslat.sh sudo ile başlar ama python'u user olarak koşar; bu yüzden user `gpio` grubunda olmalı.
- **WiFi/BT:** `yarisma_baslat.sh` rfkill ile kapatır (kılavuz 2.3 → diskalifiye nedeni). Bilgisayarsız (sadece pille) test edin.
- **NCNN:** `best_ncnn_model/` klasörü yoksa `best.pt` ile yavaş çalışır (1-3 FPS Pi4'te). NCNN ile 8-15 FPS.
- **Smoke test:** Her değişiklikten sonra `python test_smoke.py` çalıştır.
