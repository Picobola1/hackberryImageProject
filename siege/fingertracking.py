import cv2 as cv
import numpy as np
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands 
hands = mpHands.Hands()# defult parmeters

# run a web cam and anything while web cam is still going
while True:
    success, img = cap.read()

    imgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRgb)
    mpDraw = mp.solutions.drawing_utils
    #print(results.multi_hand_landmarks)
    #check if anyhands are found
    if results.multi_hand_landmarks:
        #get info from each hand/ loops thru each hand
        for handLms in results.multi_hand_landmarks:
            #not drawing on rgb img cuz we are not displaying the img
            # draw land marks of single hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            

    cv.imshow("Image", img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):   # press q to quit
        break