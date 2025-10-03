import cv2 as cv
import numpy as np
import mediapipe as mp
import random


cap = cv.VideoCapture(0)
success, img = cap.read()
window_h, window_w, c = img.shape
flappybird = cv.imread("siege/flappybird.png", cv.IMREAD_COLOR)
CoinSpawingSquareSize = 10
coinImg = cv.imread("siege/Coin.jpg", cv.IMREAD_COLOR)
mpHands = mp.solutions.hands 
hands = mpHands.Hands()# defult parmeters
resizeCoin = cv.resize(coinImg, (25,25))
coinH, coinW = resizeCoin.shape[:2]


coin = []
for i in range(5):
    CoinRangeX = random.randint(0,window_w - CoinSpawingSquareSize - 50) 
    CoinRangeY = random.randint(0,window_h - CoinSpawingSquareSize - 15)
    coin.append((CoinRangeX,CoinRangeY))
# run a web cam and anything while web cam is still going
while True:
   
    success, img = cap.read()
    flipped_img = cv.flip(img, 1) # Flip horizontally
    window_h, window_w, c = flipped_img.shape
    
    imgRgb = cv.cvtColor(flipped_img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRgb)
    mpDraw = mp.solutions.drawing_utils
    #print(results.multi_hand_landmarks)
    #check if anyhands are found
   
    #get coin width and height
    
    for (CoinRangeX, CoinRangeY) in coin:
        flipped_img[CoinRangeY:CoinRangeY+coinH, CoinRangeX:CoinRangeX+coinW] = resizeCoin

        
    #print(CoinRangeX,CoinRangeY)
    
    flipped_img[CoinRangeY:CoinRangeY + coinH, CoinRangeX:CoinRangeX + coinW] = resizeCoin
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