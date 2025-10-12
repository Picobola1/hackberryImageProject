import cv2 as cv
import mediapipe as mp
import random
import HandTrackingModule as htm

cap = cv.VideoCapture(0)
success, img = cap.read()


h, w, _ = img.shape
points = 0
flappy = cv.imread("fingerFlappy/flappybird.png")
coin_img = cv.imread("fingerFlappy/Coin.jpg")



coin_img = cv.resize(coin_img, (25, 25))
coin_h, coin_w = coin_img.shape[:2]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
detector = htm.handDetector()

tip_ids = [4, 8, 12, 16, 20]
coin_spawn_size = 10
coin_reset_cd = 0

coins = [
    (random.randint(0, w - 100), random.randint(0, h - 100))
    for _ in range(5)
]

def check_collision(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv.flip(frame, 1)
    frame = detector.findHands(frame)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)

    cv.putText(frame, str(points), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    for x, y in coins:
        if y + coin_h < h and x + coin_w < w:
            frame[y:y + coin_h, x:x + coin_w] = coin_img

    if results.multi_hand_landmarks:
        for i, hand in enumerate(results.multi_hand_landmarks):
            try:
                lm_list = detector.findPosition(frame, handNum=i, Draw=False)
            except (IndexError, AttributeError):
                continue
            if not lm_list:
                continue

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            bird_size = 45
            thumb = lm_list[tip_ids[0]][2] < lm_list[tip_ids[0] - 3][2]
            index = lm_list[tip_ids[1]][2] < lm_list[tip_ids[1] - 3][2]
            mid = lm_list[tip_ids[2]][2] < lm_list[tip_ids[2] - 3][2]
            ring = lm_list[tip_ids[3]][2] < lm_list[tip_ids[3] - 3][2]
            pinky = lm_list[tip_ids[4]][2] < lm_list[tip_ids[4] - 3][2]
            spidey = thumb and index and pinky and not mid and not ring

            for idx, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                x1, y1 = cx - bird_size // 2, cy - bird_size // 2
                if idx == 8:
                    if 0 <= x1 < w - bird_size and 0 <= y1 < h - bird_size:
                        bird_box = (x1, y1, x1 + bird_size, y1 + bird_size)
                        bird = cv.resize(flappy, (bird_size, bird_size))
                        frame[y1:y1 + bird_size, x1:x1 + bird_size] = bird

                        for j, (cx_coin, cy_coin) in enumerate(list(coins)):
                            coin_box = (cx_coin, cy_coin, cx_coin + coin_w, cy_coin + coin_h)
                            if check_collision(bird_box, coin_box):
                                del coins[j]
                                points += 1

            if spidey and coin_reset_cd == 0:
                coins = [
                    (random.randint(0, w - 100), random.randint(0, h - 100))
                    for _ in range(5)
                ]
                coin_reset_cd = 30

    if coin_reset_cd > 0:
        coin_reset_cd -= 1

    cv.imshow("Finger Flappy", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
