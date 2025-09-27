import cv2 as cv

cap = cv.VideoCapture(0)
while True:
    _, img = cap.read()
    cv.imshow('frame-1', img)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):   # press q to quit
        break

cap.release()
cv.destroyAllWindows()