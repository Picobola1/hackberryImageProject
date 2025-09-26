import cv2 as cv
import numpy as np

img = cv.imread('tiger.jpg')
## show the image with edits
cv.imshow('Tiger', img)

## making it a grey scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
## show the image with edits
cv.imshow('Gray', gray)

canny = cv.Canny(img, 125 , 175)
##         name of it     what ur chaning
cv.imshow('Canny Image', canny)

cv.waitKey(0)