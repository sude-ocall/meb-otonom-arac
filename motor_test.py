import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Pin tanımları
IN1, IN2, ENA = 17, 27, 18
IN3, IN4, ENB = 22, 23, 24

GPIO.setup([IN1, IN2, ENA, IN3, IN4, ENB], GPIO.OUT)

pwm_sol = GPIO.PWM(ENA, 1000)
pwm_sag = GPIO.PWM(ENB, 1000)
pwm_sol.start(0)
pwm_sag.start(0)

print("SOL motorlar ileri dönüyor (3 saniye)...")
GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)
pwm_sol.ChangeDutyCycle(40) # %40 hız
time.sleep(3)
pwm_sol.ChangeDutyCycle(0)

time.sleep(1)

print("SAĞ motorlar ileri dönüyor (3 saniye)...")
GPIO.output(IN3, GPIO.HIGH)
GPIO.output(IN4, GPIO.LOW)
pwm_sag.ChangeDutyCycle(40)
time.sleep(3)
pwm_sag.ChangeDutyCycle(0)

GPIO.cleanup()
print("Test bitti!")

