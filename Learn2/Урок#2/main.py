import cv2

# img = cv2.imread('images/img1.png')
# cv2.imshow('Result', img)
# cv2.waitKey(0)

#cap = cv2.VideoCapture('videos/vid1.mp4')
cap = cv2.VideoCapture(0) #для камеры
cap.set(3, 500)
cap.set(4, 300)

while True:
    success, img = cap.read()
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break