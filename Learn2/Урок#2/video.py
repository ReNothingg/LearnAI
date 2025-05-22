import cv2

cap = cv2.VideoCapture(0) #для камеры
cap.set(3, 1920/2)
cap.set(4, 1080/2)

while True:
    success, img = cap.read()
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break