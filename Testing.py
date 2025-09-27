import cv2 as cv
import numpy as np

img = cv.imread('tiger.jpg')
## show the image with edits
cv.imshow('Tiger', img)

## making it a grey scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
## show the image with edits
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)


canny = cv.Canny(blur, 125 , 175)
##         name of it     what ur chaning
cv.imshow('Canny Image', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')




cv.waitKey(0)
cv.destroyAllWindows()

