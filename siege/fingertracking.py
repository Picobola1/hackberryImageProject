import cv2 as cv
import numpy as np
import mediapipe as mp
import random


cap = cv.VideoCapture(0)
points = 0
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
    CoinRangeX = random.randint(0,window_w - CoinSpawingSquareSize - 100) 
    CoinRangeY = random.randint(0,window_h - CoinSpawingSquareSize - 100)
    coin.append((CoinRangeX,CoinRangeY))
# run a web cam and anything while web cam is still going
while True:
   
    success, img = cap.read()
    flipped_img = cv.flip(img, 1) # Flip horizontally
    window_h, window_w, c = flipped_img.shape
    
    imgRgb = cv.cvtColor(flipped_img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRgb)
    mpDraw = mp.solutions.drawing_utils
    bird_rect = None  # reset each frame

    #print(results.multi_hand_landmarks)
    #check if anyhands are found
    #get coin width and height
    #x1, y1 is minumum and + birdsize is maxium so bottem and top corner
    
    def CollisionCheck(rect1,rect2):
        x1_min,y1_min, x1_max, y1_max = rect1
        x2_min,y2_min, x2_max, y2_max = rect2
        
        return not (x1_max < x2_min or  # rect1 is left of rect2
            x1_min > x2_max or  # rect1 is right of rect2
            y1_max < y2_min or  # rect1 is above rect2
            y1_min > y2_max) 

    # Always draw coins
    for (CoinRangeX, CoinRangeY) in coin:   
        coin_rect = (CoinRangeX, CoinRangeY, CoinRangeX + coinW, CoinRangeY + coinH)
        flipped_img[CoinRangeY:CoinRangeY+coinH, CoinRangeX:CoinRangeX+coinW] = resizeCoin
    

        
    #print(CoinRangeX,CoinRangeY)

    if results.multi_hand_landmarks:
        BirdSize = 45
        #get info from each hand/ loops thru each hand
        for handLms in results.multi_hand_landmarks:
            #not drawing on rgb img cuz we are not displaying the img
            # draw land marks of single hand
            mpDraw.draw_landmarks(flipped_img, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                x1 = cx - BirdSize // 2
                y1 = cy - BirdSize // 2
                
                #print(id, cx, cy)
                if id == 8:
                    bird_rect = (x1, y1, x1 + BirdSize, y1 + BirdSize)
                    #cv.circle(flipped_img, (cx,cy), BirdSize, (255,0,255), cv.FILLED)
                    x1 = cx - BirdSize // 2
                    y1 = cy - BirdSize // 2

                    
                    resizeBird = cv.resize(flappybird, (BirdSize,BirdSize))
                    flipped_img[y1:y1+BirdSize, x1:x1+BirdSize] = resizeBird

                    for (CoinRangeX, CoinRangeY) in coin:
                        coin_rect = (CoinRangeX, CoinRangeY, CoinRangeX + coinW, CoinRangeY + coinH)
                        if CollisionCheck(bird_rect, coin_rect):
                            points +=1
                            print("hehe" + str(points))
          

    cv.imshow(" Image", flipped_img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):   # press q to quit
        break