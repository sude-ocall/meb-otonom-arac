#!/bin/bash
# yarisma_baslat.sh
# MEB Otonom Araç — Yarışma günü başlatma betiği.
#
# Kılavuz 2.3:
#   "Kontrol kartı üzerinde yer alan kızılötesi, Bluetooth, Wi-Fi veya
#    Radyo Frekansı (RF) gibi uzaktan erişim modülleri yarışma esnasında
#    kesinlikle kapalı olmalıdır. ... Tespit edilirse doğrudan diskalifiye."
#
# Bu betik:
#   1. WiFi'yi kapatır
#   2. Bluetooth'u kapatır
#   3. Sanal ortamı aktif eder
#   4. ROS2 motor node'unu arka planda başlatır
#   5. ROS2 beyin node'unu ön planda başlatır
#
# KULLANIM:
#   chmod +x yarisma_baslat.sh        # bir kez yetki ver
#   sudo ./yarisma_baslat.sh
#
# UYARI: Bu betiği çalıştırdığın anda VNC/SSH bağlantın KOPAR (WiFi kapanır).
#        Tüm ayarları VE buton bağlantısını ÖNCEDEN tamamla.
#        Bilgisayarsız sadece pille çalışacak şekilde test et!

set -e

# ── 0. Sudo kontrolü ──────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
   echo "[HATA] Bu betik root olarak çalıştırılmalı (rfkill için)."
   echo "       sudo ./yarisma_baslat.sh"
   exit 1
fi

PROJE_DIZINI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KULLANICI="${SUDO_USER:-$USER}"

echo "════════════════════════════════════════════════════════════"
echo "  MEB OTONOM ARAÇ — YARIŞMA BAŞLATMA (ROS2)"
echo "  Dizin: $PROJE_DIZINI"
echo "  Kullanıcı: $KULLANICI"
echo "════════════════════════════════════════════════════════════"

# ── 1. Kablosuz bağlantıları kapat (kılavuz 2.3 — diskalifiye nedeni) ────
# YARISMA_GUNU dosyası varsa WiFi/BT kapanır. Yoksa açık kalır (test için).
if [ -f "$PROJE_DIZINI/YARISMA_GUNU" ]; then
    echo "[1/4] YARISMA_GUNU dosyası bulundu — WiFi ve Bluetooth kapatılıyor..."
    rfkill block wifi      || echo "  uyarı: rfkill wifi başarısız"
    rfkill block bluetooth || echo "  uyarı: rfkill bluetooth başarısız"
    echo "      WiFi/Bluetooth durumu:"
    rfkill list | sed 's/^/      /'
else
    echo "[1/4] YARISMA_GUNU dosyası YOK — WiFi AÇIK kalıyor (test modu)"
fi

# ── 2. Sanal ortamı kontrol et ────────────────────────────────────────────
echo "[2/4] Sanal ortam kontrol ediliyor..."
if [ ! -d "$PROJE_DIZINI/.venv" ]; then
    echo "[HATA] $PROJE_DIZINI/.venv bulunamadı."
    echo "       Önce sanal ortamı kur: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# ── 3. Model dosyası kontrolü ─────────────────────────────────────────────
echo "[3/4] Model dosyası kontrol ediliyor..."
if [ ! -d "$PROJE_DIZINI/best_ncnn_model" ] && [ ! -f "$PROJE_DIZINI/best.pt" ] && [ ! -f "$PROJE_DIZINI/best.engine" ]; then
    echo "[HATA] best_ncnn_model/, best.pt ve best.engine bulunamadı — model yok."
    exit 1
fi
if [ ! -d "$PROJE_DIZINI/best_ncnn_model" ]; then
    echo "[UYARI] best_ncnn_model klasörü yok — best.pt ile yavaş çalışır."
    echo "        Hızlandırmak için: yolo export model=best.pt format=ncnn imgsz=320"
fi

# ── 4. Otonom aracı başlat (ROS2'suz tek dosya) ──────────────────────────
echo "[4/4] Otonom araç başlatılıyor..."
cd "$PROJE_DIZINI"

PY="$PROJE_DIZINI/.venv/bin/python3"

echo "  → baslat.py (beyin + motor tek dosya) çalıştırılıyor..."
sudo -E -u "$KULLANICI" "$PY" baslat.py

echo "════════════════════════════════════════════════════════════"
echo "  YARIŞMA SONLANDI"
echo "════════════════════════════════════════════════════════════"
