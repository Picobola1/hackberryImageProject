import cv2 as cv
import numpy as np
import mediapipe as mp
import random
import HandTrackingModule as htm

cap = cv.VideoCapture(0)
points = 0
success, img = cap.read()
window_h, window_w, c = img.shape
flappybird = cv.imread("flappybird.png", cv.IMREAD_COLOR)
CoinSpawingSquareSize = 10
coinImg = cv.imread("fingerFlappy/Coin.jpg", cv.IMREAD_COLOR)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
resizeCoin = cv.resize(coinImg, (25, 25))
coinH, coinW = resizeCoin.shape[:2]
tipIds = [4, 8, 12, 16, 20]
detector = htm.handDetector()

coin = []
for i in range(5):
    CoinRangeX = random.randint(0, window_w - CoinSpawingSquareSize - 100)
    CoinRangeY = random.randint(0, window_h - CoinSpawingSquareSize - 100)
    coin.append((CoinRangeX, CoinRangeY))

coin_reset_cooldown = 0

def CollisionCheck(rect1, rect2):
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

while True:
    success, img = cap.read()
    flipped_img = cv.flip(img, 1)
    window_h, window_w, c = flipped_img.shape
    imgRgb = cv.cvtColor(flipped_img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRgb)
    bird_rect = None
    cv.putText(flipped_img, str(points), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    for (CoinRangeX, CoinRangeY) in coin:
        if CoinRangeY + coinH < window_h and CoinRangeX + coinW < window_w:
            flipped_img[CoinRangeY:CoinRangeY + coinH, CoinRangeX:CoinRangeX + coinW] = resizeCoin

    flipped_img = detector.findHands(flipped_img)
    lmList = detector.findPosition(flipped_img, Draw=False)

    if len(lmList) != 0 and results.multi_hand_landmarks:
        BirdSize = 45
        for i, handLms in enumerate(results.multi_hand_landmarks):
            lmList = detector.findPosition(flipped_img, handNum=i, Draw=False)
            thumb_open = lmList[tipIds[0]][2] < lmList[tipIds[0] - 3][2]
            index_open = lmList[tipIds[1]][2] < lmList[tipIds[1] - 3][2]
            middle_open = lmList[tipIds[2]][2] < lmList[tipIds[2] - 3][2]
            ring_open = lmList[tipIds[3]][2] < lmList[tipIds[3] - 3][2]
            pinky_open = lmList[tipIds[4]][2] < lmList[tipIds[4] - 3][2]
            spidey_pose = thumb_open and index_open and pinky_open and not middle_open and not ring_open
            mpDraw.draw_landmarks(flipped_img, handLms, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = flipped_img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x1 = cx - BirdSize // 2
                y1 = cy - BirdSize // 2

                if id == 8:
                    bird_rect = (x1, y1, x1 + BirdSize, y1 + BirdSize)
                    resizeBird = cv.resize(flappybird, (BirdSize, BirdSize))
                    if 0 <= y1 < window_h - BirdSize and 0 <= x1 < window_w - BirdSize:
                        flipped_img[y1:y1 + BirdSize, x1:x1 + BirdSize] = resizeBird

                    for j, (CoinRangeX, CoinRangeY) in enumerate(list(coin)):
                        coin_rect = (CoinRangeX, CoinRangeY, CoinRangeX + coinW, CoinRangeY + coinH)
                        if CollisionCheck(bird_rect, coin_rect):
                            del coin[j]
                            points += 1

            if spidey_pose and coin_reset_cooldown == 0:
                coin = []
                for i in range(5):
                    CoinRangeX = random.randint(0, window_w - CoinSpawingSquareSize - 100)
                    CoinRangeY = random.randint(0, window_h - CoinSpawingSquareSize - 100)
                    coin.append((CoinRangeX, CoinRangeY))
                coin_reset_cooldown = 30

    if coin_reset_cooldown > 0:
        coin_reset_cooldown -= 1

    cv.imshow("Image", flipped_img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
