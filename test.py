import numpy as np
import cv2 as cv
import glob

img = cv.imread('vehicle_speed_estimation/undistort.jpeg')
images = glob.glob('vehicle_speed_estimation/imgs/calibration3/*.jpeg')

cv.imshow('img', img)
cv.waitkey(100)
cv.destroyAllWindows()
'''
for image in images:
    cv.imshow('img', image)
cv.waitKey(100)
cv.destroyAllWindows()
'''