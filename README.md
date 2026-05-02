# MEB Robot Yarışması — Otonom Araç

18\. Uluslararası MEB Robot Yarışması, Otonom Araç kategorisi için geliştirilmiş proje.
Araç, kamera görüntüsünden şerit takibi yapar; tabelaları ve trafik ışıklarını tanıyarak görevleri tamamlar.

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `main.py` | Ana yarışma döngüsü. Tüm modülleri bir araya getirir; görev sıralamasını ve motor komutlarını yönetir. |
| `nesne_beyni.py` | YOLOv8 ile nesne/tabela/ışık tespiti. `best.pt` modelini yükler; güven eşiği, HSV doğrulama ve sınıf sorgulama fonksiyonlarını içerir. |
| `serit_beyni.py` | Perspektif dönüşümü + histogram ile şerit takibi. Direksiyon sapma açısını döndürür. |
| `motor_surucu.py` | L298N motor sürücüsü için GPIO fonksiyonları (`ileri_git`, `dur`, `direksiyon_cevir`). |
| `tabela_beyni.py` | OpenCV tabanlı yedek tabela tanıma (şekil analizi). Ana akışta kullanılmıyor; test amaçlı. |
| `ros2_beyin_yayin.py` | ROS2 publisher node. `/arac_komut` ve `/serit_sapma` topic'lerine komut yayınlar. |
| `ros2_motor_dinleyici.py` | ROS2 subscriber node. Yayınlanan komutları dinler ve motor pinlerine uygular. |
| `test_kamera.py` | MacBook'ta simülasyon testi. Motor sinyali göndermez; terminale log basar. |
| `test_tespit.py` | Canlı tespit tanı aracı. Kamera görüntüsünde tüm tespitleri gösterir; `+`/`-` ile eşiği ayarla. |
| `best.pt` | YOLOv8 model ağırlıkları (26 sınıf). Proje kök dizinine koy. |
| `requirements.txt` | Python bağımlılıkları. |

---

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Çalıştırma

### Yarışma modu (Raspberry Pi / Jetson)
```bash
python main.py
```

### MacBook simülasyonu (motor sinyali yok)
```bash
python test_kamera.py
```

### Canlı tespit tanısı (kameraya tabela tut, tespitleri gör)
```bash
python test_tespit.py
```

### ROS2 modu
```bash
# Terminal 1 — beyin
ros2 run otonom_arac ros2_beyin_yayin

# Terminal 2 — motor
ros2 run otonom_arac ros2_motor_dinleyici
```

---

## Model (`best.pt`)

- Proje kök dizinine (`MEB_Otonom_Arac/`) koy.
- Güven eşiği `nesne_beyni.py` içindeki `GUVEN_ESIGI` sabiti ile ayarlanır (varsayılan: `0.35`).
  - Çok yanlış tespit → `0.45`'e çek
  - Tespit yok → `0.25`'e düşür

---

## Notlar

> **Alara için:**
> Kamera açısını hem şeridi hem tabelayı görecek şekilde ayarla.
> Şerit takibi için zemin çizgileri, tabela tespiti için tabela yüzleri kamera görüş alanında olmalı.
> Test için `test_tespit.py` aracını kullan — canlı görüntüde tespitleri ve confidence değerlerini gösterir.

> **Emir için:**
> Motor pinlerini `ros2_motor_dinleyici.py` içindeki `# TODO` kısımlarına gir.
> L298N bağlantısı için hazır pin sabitleri dosyanın üstünde yorum olarak bırakıldı:
> `IN1=17, IN2=27, ENA=18` (sol motor) · `IN3=22, IN4=23, ENB=24` (sağ motor) · Servo: `pin 12`
> GPIO kurulumunu `motorlari_hazirla()` fonksiyonuna ekle; `ileri_git`, `dur`, `direksiyon_cevir` fonksiyonları zaten iskelet olarak yazıldı.

---

## Görev Özeti

| Görev | Tabela / Durum | Davranış |
|---|---|---|
| 1 | Yeşil trafik ışığı | Yeşil görene kadar dur; görünce hareket başla |
| 2 | Yaya geçidi tabelası | 5 saniye dur, devam et |
| 3 | Hız tümseği tabelası | Hızı düşür, geç, normal hıza dön |
| 4 | Hemzemin geçit tabelası | 5 saniye dur, devam et |
| 5 | Sollama serbest tabelası | Sola kayarak sollamayı tamamla, kendi şeridine dön |
| 6 | Çıkmaz yol tabelası | Dur, sağa dön, yola devam et |
| 7 | Park tabelası → kırmızı park alanı | Park alanını bul, park et, yarışmayı bitir |
