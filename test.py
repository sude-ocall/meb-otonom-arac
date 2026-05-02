import cv2

# İndeksi önce 1, sonra 0 veya 2 olarak dene
cap = cv2.VideoCapture(1) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera okunamadı, indeks yanlış olabilir!")
        break
    
    cv2.imshow("MacBook Kamera Testi", frame)
    
    # Kapatmak için klavyeden 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()