import cv2 as cv
import numpy as np

pizza = cv.imread('pizza.jpg')
img = cv.imread('1.jpg')

def translatePizza(pizza, x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    demensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(pizza, transMat,demensions)

translated = translatePizza(pizza, 100, 100)

cv.imshow('translated pizza', translated)
cv.imshow('pizza', pizza)
## show the image with edits



gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


haar_cascadeFaces = cv.CascadeClassifier('haar_face.xml')
#detect a face and return rectangluar coordinates as a list
rect_around_face = haar_cascadeFaces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f'Number Of Faces Found = {len(rect_around_face)}')


for (x,y,w,h) in rect_around_face:
   
    
    cv.rectangle(img, (x,y), (x+w,y + h), (100,100,0), thickness=2)



cv.imshow('Detected Faces', img)






cv.waitKey(0)
cv.destroyAllWindows()

