import cv2
import numpy as np

photo = np.zeros((450, 450, 3), dtype='uint8')

#RGB - based
#BGR - в OpenCV
photo[100:150, 200:280] = 255, 12, 0

cv2.imshow('Photo', photo)
cv2.waitKey(0)