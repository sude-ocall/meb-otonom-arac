#!/bin/bash
# otobaslat_kur.sh
# ─────────────────────────────────────────────────────────────────
# Bu scripti Pi üzerinde BİR KEZ çalıştırıyorsun.
# Bundan sonra Pi'ye güç verildiği anda yarışma kodu otomatik başlar.
#
# KULLANIM:
#   chmod +x otobaslat_kur.sh
#   sudo ./otobaslat_kur.sh
# ─────────────────────────────────────────────────────────────────

set -e

# ── Sudo kontrolü ─────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
   echo "[HATA] sudo ile çalıştır:  sudo ./otobaslat_kur.sh"
   exit 1
fi

PROJE_DIZINI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KULLANICI="${SUDO_USER:-pi}"

echo "════════════════════════════════════════════════════════════"
echo "  OTOMATİK BAŞLATMA KURULUMU"
echo "  Proje dizini: $PROJE_DIZINI"
echo "  Kullanıcı: $KULLANICI"
echo "════════════════════════════════════════════════════════════"

# ── 1. Servis dosyasını oluştur ───────────────────────────────
SERVIS_DOSYASI="/etc/systemd/system/meb-otonom.service"

cat > "$SERVIS_DOSYASI" << EOF
[Unit]
Description=MEB Otonom Arac - Yarisma Otomatik Baslatma
After=multi-user.target
Wants=multi-user.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJE_DIZINI
ExecStartPre=/bin/sleep 5
ExecStart=/bin/bash $PROJE_DIZINI/yarisma_baslat.sh
Restart=on-failure
RestartSec=3
StandardOutput=append:/var/log/meb-otonom.log
StandardError=append:/var/log/meb-otonom.log
Environment=HOME=/home/$KULLANICI
Environment=SUDO_USER=$KULLANICI

[Install]
WantedBy=multi-user.target
EOF

echo "[1/3] Servis dosyası oluşturuldu: $SERVIS_DOSYASI"

# ── 2. Servisi aktif et ───────────────────────────────────────
systemctl daemon-reload
systemctl enable meb-otonom.service
echo "[2/3] Servis aktif edildi (her açılışta çalışacak)"

# ── 3. Log dosyasını hazırla ──────────────────────────────────
touch /var/log/meb-otonom.log
chown "$KULLANICI":"$KULLANICI" /var/log/meb-otonom.log
echo "[3/3] Log dosyası hazırlandı: /var/log/meb-otonom.log"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ KURULUM TAMAMLANDI!"
echo ""
echo "  Şimdi ne olacak:"
echo "    → Pi her açıldığında 5 saniye bekler"
echo "    → WiFi ve Bluetooth otomatik kapanır"
echo "    → Motor ve beyin node'ları otomatik başlar"
echo "    → Buton beklemeye geçer (butona bas = yarışma başlar)"
echo ""
echo "  FAYDALI KOMUTLAR:"
echo "    Durumu gör:     sudo systemctl status meb-otonom"
echo "    Logları gör:    cat /var/log/meb-otonom.log"
echo "    Elle başlat:    sudo systemctl start meb-otonom"
echo "    Elle durdur:    sudo systemctl stop meb-otonom"
echo "    İPTAL ET:       sudo systemctl disable meb-otonom"
echo "════════════════════════════════════════════════════════════"
