import cv2 as cv
import numpy as np
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
flappybird = cv.imread("siege/flappybird.png", cv.IMREAD_COLOR)
mpHands = mp.solutions.hands 
hands = mpHands.Hands()# defult parmeters

# run a web cam and anything while web cam is still going
while True:
    success, img = cap.read()
    flipped_img = cv.flip(img, 1) # Flip horizontally
    imgRgb = cv.cvtColor(flipped_img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRgb)
    mpDraw = mp.solutions.drawing_utils
    #print(results.multi_hand_landmarks)
    #check if anyhands are found
    if results.multi_hand_landmarks:
        #get info from each hand/ loops thru each hand
        for handLms in results.multi_hand_landmarks:
            #not drawing on rgb img cuz we are not displaying the img
            # draw land marks of single hand
            mpDraw.draw_landmarks(flipped_img, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                if id == 8:
                    BirdSize = 45
                    #cv.circle(flipped_img, (cx,cy), BirdSize, (255,0,255), cv.FILLED)
                    x1 = cx - BirdSize // 2
                    y1 = cy - BirdSize // 2

                    resizeBird = cv.resize(flappybird, (BirdSize,BirdSize))
                    flipped_img[y1:y1+BirdSize, x1:x1+BirdSize] = resizeBird
                    

    cv.imshow(" Image", flipped_img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):   # press q to quit
        break