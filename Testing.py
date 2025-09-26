import cv2 as cv
import numpy as np

img = cv.imread('tiger.jpg')

cv.imshow('Tiger', img)


cv.waitKey(0)