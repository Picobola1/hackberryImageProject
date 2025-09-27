import cv2 as cv
import numpy as np

img = cv.imread('1.jpg')
## show the image with edits
cv.imshow('Me', img)





cv.waitKey(0)
cv.destroyAllWindows()

