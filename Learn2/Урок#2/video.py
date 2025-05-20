import cv2

cap = cv2.VideoCapture(0) #для камеры
cap.set(3, 500)
cap.set(4, 300)

while True:
    success, img = cap.read()
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break