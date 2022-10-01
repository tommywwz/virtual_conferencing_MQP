import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pg

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)  # width
cam.set(4, 360)  # height
# image = cv2.imread("vid/single_user.jpg")
old_lines = None
while cam.isOpened():

    success, image = cam.read()

    if success:
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (3, 3), 0)
        edged = cv2.Canny(blurred, 150, 160)
        # line_image = np.copy(image) * 0
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 15, np.array([]), 90, 0)
        # old_lines = lines
        line_image = image.copy()

        if lines is not None:
            print(lines)
            old_lines = lines.copy()
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if np.abs(y1 - y2) < 25:

                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        elif old_lines is not None:
            for line in old_lines:
                # image = cv2.polylines(line_image, [old_lines],
                #                       False, (0, 255, 255), 2)
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # line_image = cv2.addWeighted(image, 1, line_image, 1, 0)
        cv2.imshow("Original image", image)
        cv2.imshow("Edged image", edged)
        cv2.imshow("Lined image", line_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
