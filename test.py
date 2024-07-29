import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgSize = 300

labels = ["Strength", "Victory", "Like", "Dislike", "Okay", "Callme", "Enjoy"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break  # Exit the loop if failed to read from the camera

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the hand bounding box stays within the image dimensions
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])

        # Check if hand bounding box is valid
        if x2 > x1 and y2 > y1:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y1:y2, x1:x2]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)
            

            cv2.imshow("imagecrop", imgCrop)
            cv2.imshow("imagewhite", imgWhite)

    cv2.imshow("image", img)

    # Exit program on pressing the escape key
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
