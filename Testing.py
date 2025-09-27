import cv2 as cv
import numpy as np

img = cv.imread('1.jpg')
## show the image with edits
cv.imshow('Me', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('GrayMe', gray)




cv.waitKey(0)
cv.destroyAllWindows()

