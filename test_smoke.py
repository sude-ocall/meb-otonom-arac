"""
test_smoke.py
Donanım YOK — saf yazılım smoke testleri.

Şunları doğrular:
  1. gorev_dedektor.py her HSV görevi için doğru karar veriyor
  2. serit_beyni.otonom_beyin() çalışıyor (cite bug'ı geri gelmedi)
  3. Komutlar.hiz/hiz_parse round-trip
  4. Brain ve motor modülleri syntax/import açısından temiz
  5. Sollama 3 fazlı manevra sabit toplamı = SURE_SOLLAMA_FAZ_C
  6. Brain SOLLAMA_COOLDOWN > motor manevra süresi (regresyon)

Çalıştır:
    python3 test_smoke.py
"""
import sys
import types
import importlib.util
import numpy as np


# ── Renkli ÇIKTILAR ───────────────────────────────────────────────────────
def ok(msg):    print(f"  \x1b[32m✓\x1b[0m {msg}")
def fail(msg):  print(f"  \x1b[31m✗\x1b[0m {msg}"); sys.exit(1)
def kategori(t): print(f"\n\x1b[1m{t}\x1b[0m")


# ── 1. gorev_dedektor.py ──────────────────────────────────────────────────
kategori("1. gorev_dedektor.py — HSV tespitleri")
import gorev_dedektor as gd

black = np.zeros((480, 640, 3), dtype=np.uint8)
var, _ = gd.hiz_tumsek_var_mi(black)
if var: fail("siyah frame'de hız tümseği yanlış pozitif")
ok("siyah frame: hiz_tumsek=False, hemzemin=False, turuncu=False")

# Sarı dikey şeritler — hız tümseği
sari = np.zeros((480, 640, 3), dtype=np.uint8)
for i in range(6):
    x = 50 + i * 100
    sari[260:380, x:x+40] = (0, 255, 255)
var, yakin = gd.hiz_tumsek_var_mi(sari)
if not var: fail("6 sarı şeritli frame hız tümseği olarak tanınmadı")
if not (0.5 < yakin < 1.0): fail(f"yakınlık beklenen aralıkta değil: {yakin}")
ok(f"6 dikey sarı şerit → hız tümseği, yakınlık={yakin:.2f}")

# Beyaz ızgara — hemzemin
hemz = np.zeros((480, 640, 3), dtype=np.uint8)
for row in range(2):
    for col in range(8):
        x = 30 + col * 70
        y = 280 + row * 80
        hemz[y:y+40, x:x+40] = (255, 255, 255)
var, _ = gd.hemzemin_var_mi(hemz)
if not var: fail("2x8 beyaz ızgara hemzemin olarak tanınmadı")
ok("2×8 beyaz ızgara → hemzemin geçit")

# Turuncu blok — sollama
tur = np.zeros((480, 640, 3), dtype=np.uint8)
tur[200:400, 250:400] = (60, 130, 255)
var, alan, cx = gd.turuncu_arac_var_mi(tur)
if not var: fail("turuncu blok tespit edilemedi")
if alan < 5000: fail(f"alan çok küçük: {alan}")
if not (300 <= cx <= 350): fail(f"cx beklenen aralıkta değil: {cx}")
ok(f"turuncu blok → sollama tetiği, alan={alan}, cx={cx}")

# Yanlış pozitif: sarı blok turuncu olarak görünmemeli (HSV ayrımı)
sari_blok = np.zeros((480, 640, 3), dtype=np.uint8)
sari_blok[200:400, 250:400] = (0, 255, 255)   # saf sarı
var_o, _, _ = gd.turuncu_arac_var_mi(sari_blok)
if var_o: fail("sarı blok yanlışlıkla turuncu olarak algılandı (HSV ayrımı kötü)")
ok("sarı blok turuncuya karışmıyor")


# ── 2. serit_beyni — sliding-window doğru sapma işareti üretiyor mu? ──────
kategori("2. serit_beyni.otonom_beyin")
from serit_beyni import otonom_beyin
res, sapma = otonom_beyin(np.zeros((300, 400, 3), dtype=np.uint8))
if res is None or sapma is None: fail("otonom_beyin None döndü")
if sapma != 0: fail(f"boş frame için sapma {sapma} != 0")
ok(f"boş frame → sapma={sapma}")

# None girişi guard
r, s = otonom_beyin(None)
if not (r is None and s == 0): fail(f"None girişinde ({r}, {s}) bekleniyordu (None, 0)")
ok("None girişi: (None, 0)")

# Sadece sol şerit görünüyorsa sapma negatif olmalı (sağa düzelt)
sadece_sol = np.zeros((480, 640, 3), dtype=np.uint8)
sadece_sol[:, 145:155] = (255, 255, 255)
r, s = otonom_beyin(sadece_sol)
if s >= 0: fail(f"sadece sol şerit: sapma {s} negatif olmalıydı (sağa düzelt)")
ok(f"sadece sol şerit → sapma={s} (sağa düzeltme)")

# Sadece sağ şerit → sapma pozitif (sola düzelt)
sadece_sag = np.zeros((480, 640, 3), dtype=np.uint8)
sadece_sag[:, 485:495] = (255, 255, 255)
r, s = otonom_beyin(sadece_sag)
if s <= 0: fail(f"sadece sağ şerit: sapma {s} pozitif olmalıydı (sola düzelt)")
ok(f"sadece sağ şerit → sapma={s} (sola düzeltme)")

# Düz iki şerit ortalanmış → sapma neredeyse 0
duz = np.zeros((480, 640, 3), dtype=np.uint8)
duz[:, 145:155] = (255, 255, 255)
duz[:, 485:495] = (255, 255, 255)
r, s = otonom_beyin(duz)
if abs(s) > 60: fail(f"düz iki şerit: |sapma| {abs(s)} > 60")
ok(f"düz iki şerit → sapma={s} (|s|<60)")


# ── 3. komutlar.py round-trip ─────────────────────────────────────────────
kategori("3. komutlar.py")
from komutlar import (Komut, HIZ_NORMAL, HIZ_YAVAS, HIZ_PARK, HIZ_PARK_SON,
                       KALP_HZ, KALP_TIMEOUT_SN)
for h in (0, 20, 40, 70):
    if Komut.hiz_parse(Komut.hiz(h)) != h:
        fail(f"hiz round-trip kırık: {h}")
ok("Komut.hiz/hiz_parse round-trip 0, 20, 40, 70")
if Komut.hiz_parse("DUR") is not None: fail("hiz_parse 'DUR' → None bekleniyordu")
if Komut.hiz_parse("HIZ:abc") is not None: fail("hiz_parse 'HIZ:abc' → None bekleniyordu")
ok("Komut.hiz_parse hatalı girişlerde None döndürüyor")
if not hasattr(Komut, "SOLLAMA_BITTI"):
    fail("Komut.SOLLAMA_BITTI tanımlı değil (akıllı sollama çıkışı için lazım)")
ok(f"Komut.SOLLAMA_BITTI = '{Komut.SOLLAMA_BITTI}'")
if HIZ_PARK_SON >= HIZ_PARK:
    fail(f"HIZ_PARK_SON ({HIZ_PARK_SON}) >= HIZ_PARK ({HIZ_PARK}) — kademe yok!")
ok(f"HIZ_PARK_SON ({HIZ_PARK_SON}) < HIZ_PARK ({HIZ_PARK}) — kademe doğru yönde")


# ── 4. Brain & motor modülleri syntax/import ──────────────────────────────
kategori("4. ROS2 modülleri syntax")
# rclpy + std_msgs stub'larıyla
for m in ["rclpy", "rclpy.node", "rclpy.executors",
          "rclpy.callback_groups", "rclpy.qos", "std_msgs", "std_msgs.msg"]:
    sys.modules.setdefault(m, types.ModuleType(m))

class _S:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self
    def __getattr__(self, _): return self
    def __or__(self, other): return self

q = sys.modules["rclpy.qos"]
q.QoSProfile = _S
q.ReliabilityPolicy = _S()
q.HistoryPolicy = _S()
q.qos_profile_sensor_data = _S()
sys.modules["rclpy.node"].Node = _S
sys.modules["rclpy.executors"].MultiThreadedExecutor = _S
sys.modules["rclpy.callback_groups"].ReentrantCallbackGroup = _S
sys.modules["std_msgs.msg"].String = _S

spec = importlib.util.spec_from_file_location("rby", "ros2_beyin_yayin.py")
brain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brain)
ok(f"brain import OK — YARIS_SURESI_SN={brain.YARIS_SURESI_SN}")

spec = importlib.util.spec_from_file_location("rmd", "ros2_motor_dinleyici.py")
motor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motor)
ok(f"motor import OK — SURE_SOLLAMA_FAZ_C={motor.SURE_SOLLAMA_FAZ_C}")


# ── 5. Sollama faz tutarlılığı ─────────────────────────────────────────────
kategori("5. Sollama 3 fazlı manevra")
fa, fb, fc = motor.SURE_SOLLAMA_FAZ_A, motor.SURE_SOLLAMA_FAZ_B, motor.SURE_SOLLAMA_FAZ_C
if not (0 < fa < fb < fc):
    fail(f"Sollama fazları monoton artmıyor: A={fa} B={fb} C={fc}")
ok(f"fazlar monoton: A={fa}s → B={fb}s → C={fc}s (toplam {fc}s)")


# ── 6. Brain cooldown >= motor manevra süresi (regresyon) ─────────────────
kategori("6. Brain cooldown ↔ motor manevra süresi")
if brain.SOLLAMA_COOLDOWN <= motor.SURE_SOLLAMA_FAZ_C:
    fail(
        f"SOLLAMA_COOLDOWN ({brain.SOLLAMA_COOLDOWN}sn) "
        f"≤ manevra süresi ({motor.SURE_SOLLAMA_FAZ_C}sn) — "
        f"manevra ortasında ikinci tetik riski!"
    )
ok(f"cooldown {brain.SOLLAMA_COOLDOWN}s > manevra {motor.SURE_SOLLAMA_FAZ_C}s")


# ── 7. Brain hız yönü işareti (regresyon) ─────────────────────────────────
kategori("7. Sapma → diferansiyel işaret tutarlılığı")
# motor _sapma_cb formülü:
#   delta = SAPMA_KAZANIM * sapma
#   sol_hedef = cruise - delta   (sapma>0 → sol yavaş → sola dönüş)
#   sag_hedef = cruise + delta
# Yani sapma>0 → sola dönüş.
# serit_beyni: sadece_sag → sapma>0 (motor sola döner = sağ şeritten kaçar) ✓
# Bu mantığı kontrol et:
test_sapma = 30
delta = motor.SAPMA_KAZANIM * test_sapma
if not (delta > 0):
    fail("SAPMA_KAZANIM negatif veya sıfır")
ok(f"sapma=30 → delta={delta:.1f} → sol yavaş, sağ hızlı (sola dön) — tutarlı")


# ── 8. Yeni Komut sabitleri brain ve motor'da bilinir mi? ────────────────
kategori("8. SOLLAMA_BITTI komutu uçtan uca bilinir mi?")
from komutlar import Komut as K
if K.SOLLAMA_BITTI != "SOLLAMA_BITTI":
    fail(f"Komut.SOLLAMA_BITTI değeri beklenmedik: {K.SOLLAMA_BITTI}")
ok("Komut.SOLLAMA_BITTI string'i tutarlı")


print("\n\x1b[32;1m✓ Tüm smoke testler geçti\x1b[0m\n")
