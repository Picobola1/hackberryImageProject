import cv2 as cv
import numpy as np
import mediapipe as mp
import random
import time


class handDetector():
    def __init__(self, mode=False,maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        # make object and each object will have its own variable
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
            )# defult parmeters
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw = True):

        imgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)

        if self.results.multi_hand_landmarks:
        #get info from each hand/ loops thru each hand
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
        return img
    def findPosition(self,img, handNum=0, Draw = True):
        lmList  = []
        if self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id,cx,cy])
                    if Draw:
                        cv.circle(img, (cx,cy), 6, (100,200,255), cv.FILLED)
                    
        return lmList


    

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[8])
 
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img,str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 )
        cv.imshow(" Image", img)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):   # press q to quit
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
