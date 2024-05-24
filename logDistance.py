from cameraCalibration import removeDistortion, calibrate
import cv2 as cv

lpCmWidth = 40
lpCmHeight = 13
realDistance = 103#distance in cm from lp1.jpeg
lpPWidth = 582#527distorted#measurement in pixels from lp1.jpeg
lpPHeight = 174#measurement in pixels from lp1.jpeg

fLength = (lpPWidth * realDistance) / lpCmWidth

distance = (lpCmWidth * fLength) / lpPWidth

#DISTORTION HANDLING

camMatrix, distCoeff = calibrate(showPics=False)
undistortImg = removeDistortion(camMatrix, distCoeff, 'vehicle_speed_estimation/imgs/lpImgs/lp1.jpeg', showpics=False)

cv.imwrite('vehicle_speed_estimation/imgs/undistortedImage.jpeg', undistortImg)

undistortImg3 = removeDistortion(camMatrix, distCoeff, 'vehicle_speed_estimation/imgs/lpImgs/lp3.jpeg', showpics=False)
cv.imwrite('vehicle_speed_estimation/imgs/undistortedImage3.jpeg', undistortImg3)

lpPWidth = 191
lpPHeight = 56.5

print((lpCmWidth * fLength) / lpPWidth)

undistortImg10 = removeDistortion(camMatrix, distCoeff, 'vehicle_speed_estimation/imgs/lpImgs/lp10.jpeg', showpics=False)
cv.imwrite('vehicle_speed_estimation/imgs/undistortedImage10.jpeg', undistortImg10)

lpPWidth = 57
print((lpCmWidth * fLength) / lpPWidth)


