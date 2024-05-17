import numpy as np
import cv2 as cv

img = cv.imread('imgs/undistort.jpeg')
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()