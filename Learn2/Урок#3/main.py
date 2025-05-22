import cv2
import numpy as np

photo = np.zeros((450, 450, 3), dtype='uint8')

#RGB - based
#BGR - в OpenCV
#photo[100:150, 200:280] = 255, 12, 0

cv2.rectangle(photo, (50, 70), (100, 100), (255, 0, 0), thickness=cv2.FILLED) #
cv2.line(photo, (0, photo.shape[0] // 2), (photo.shape[1], photo.shape[0] // 2), (255, 0, 0), thickness=3)
cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 100, (255, 0, 0), thickness=3)

cv2.putText(photo, 'Text', (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), thickness=2)

cv2.imshow('Photo', photo)
cv2.waitKey(0)
