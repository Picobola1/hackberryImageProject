import cv2 as cv
import numpy as np
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

# run a web cam
while True:
    success, img = cap.read()

    cv.imshow("Image", img)
    cv.waitKey(1)