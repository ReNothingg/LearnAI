import cv2
import numpy as np

img = cv2.imread('images/img1.png')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

#cv2.imshow('Result (resize)', img)
#cv2.imshow('Result (resize + shape)', new_img[0:100, 0:150])
#print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (9, 9), 0)

img = cv2.Canny(img, 200, 200)

kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

img = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Result', img)
cv2.waitKey(0)