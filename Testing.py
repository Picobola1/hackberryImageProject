import cv2 as cv
import numpy as np

img = cv.imread('5.jpg')
## show the image with edits



gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


haar_cascadeFaces = cv.CascadeClassifier('haar_face.xml')
#detect a face and return rectangluar coordinates as a list
rect_around_face = haar_cascadeFaces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f'Number Of Faces Found = {len(rect_around_face)}')


for (x,y,w,h) in rect_around_face:
    
    cv.rectangle(img, (x,y), (x+w,y + h), (0,250,0), thickness=2)



cv.imshow('Detected Faces', img)






cv.waitKey(0)
cv.destroyAllWindows()

