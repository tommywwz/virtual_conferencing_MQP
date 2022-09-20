import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pg

image = cv2.imread("vid/single_user.jpg")
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (3, 3), 0)
edged = cv2.Canny(blurred, 150, 160)
line_image = np.copy(image) * 0
lines = cv2.HoughLinesP(edged, 1, np.pi/180, 15, np.array([]), 90, 0)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

line_image = cv2.addWeighted(image, 1, line_image, 1, 0)

cv2.imshow("Original image", image)
cv2.imshow("Edged image", edged)
cv2.imshow("Lined image", line_image)
cv2.waitKey(0)
