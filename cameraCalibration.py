import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

def calibrate(showPics = True):
    images = glob.glob('vehicle_speed_estimation/imgs/calibration/*.jpeg')
    cboardSize = (7, 7)#rows and columns
    framesize = (1600, 1200)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #create world points matrix that are going to be projected on
    worldPts = np.zeros((cboardSize[0] * cboardSize[0], 3), np.float32)
    #these are placeholder numbers for the world points
    worldPts[:, :2] = np.mgrid[0:cboardSize[0], 0:cboardSize[1]].T.reshape(-1, 2)

    worldPtsList = []
    imgPtsList = []

    #find corners
    for image in images:
        img = cv.imread(image)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #find chessboard corners. The method only takes grayscale images
        foundCorners, corners = cv.findChessboardCorners(imgGray, cboardSize, None)

        if foundCorners == True:
            worldPtsList.append(worldPts)
            #cornerSubPix finds a more accurate locatino of the corners
            cornersRefined = cv.cornerSubPix(imgGray, corners, (11,11), (-1,-1), criteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(img, cboardSize, cornersRefined, foundCorners)
                img = cv.resize(img, (960, 540)) 
                cv.imshow('chessboard', img)
                cv.waitKey(0)
    cv.destroyAllWindows()

    #calibration
    repError, camMatrix, distCoeff, rvec, tvec = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    print('Camera matrix:\n', camMatrix)
    print('reprojection error: {:.4f}'.format(repError))

    return camMatrix, distCoeff

def removeDistortion(camMatrix, distCoeff):
    img = cv.imread('vehicle_speed_estimation/imgs/undistort.jpeg')
    height, width = img.shape[:2]
    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (width, height), 0, (width, height))
    imgUndist = cv.undistort(img, camMatrix, distCoeff, None, camMatrixNew)

    #draw line to see distortion changes
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()

def runRemoveDistortion():
    camMatrix, distCoeff = calibrate(showPics = False)
    removeDistortion(camMatrix, distCoeff)

if __name__ == '__main__':
    #calibrate()  
    runRemoveDistortion()