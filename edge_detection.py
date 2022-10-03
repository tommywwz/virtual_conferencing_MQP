import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pg

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(3, 640)  # width
cam.set(4, 360)  # height
rawimage = cv2.imread("vid/test1.jpg")
rawimage = cv2.resize(rawimage.copy(), (640, 480))
stored_lines = {}
lines_denoised = []
counter = 0

while cam.isOpened():
    success, rawimage = cam.read()

# while rawimage is not None:
#     success = True

    if success:
        rawimage = cv2.rotate(rawimage, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("raw", rawimage)
        h, w, c = rawimage.shape
        image = rawimage[int(h / 2):h, :, :]
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        im = cv2.filter2D(image, -1, kernel)
        cv2.imshow("Sharpening", im)
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (21, 7), 0)
        edged = cv2.Canny(blurred, threshold1=10, threshold2=50)
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180,
                                threshold=15, lines=np.array([]), minLineLength=80, maxLineGap=3)
        # stored_lines = lines
        line_image = image.copy()

        if counter > 30:
            counter = 0
            # max_key = max(len(item) for item in stored_lines.values())
            max_key = max(stored_lines, key=lambda x:len(stored_lines[x]))
            lines_denoised = stored_lines.get(max_key)
            stored_lines.clear()

        if lines is not None:
            # print(lines)
            # stored_lines.clear()
            for line in lines:
                for x1, y1, x2, y2 in line:
                    dy = y1 - y2
                    dx = x1 - x2
                    if np.abs(dy) < np.abs(dx)*0.12:
                        # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        slope = round(100*dy/dx)
                        if stored_lines.get(slope) is None:
                            stored_lines[slope] = [line]
                        else:
                            stored_lines[slope].append(line)
                        print(stored_lines)
                        counter += 1

        if lines_denoised:
            for line_denoised in lines_denoised:
                for x1, y1, x2, y2 in line_denoised:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # elif stored_lines:
        #     print(stored_lines)
        #     for lines in stored_lines.values():
        #         for line in lines:
        #             for x1, y1, x2, y2 in line:
        #                 cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)



        # line_image = cv2.addWeighted(image, 1, line_image, 1, 0)
        cv2.imshow("Original image", image)
        cv2.imshow("Blured image", blurred)
        cv2.imshow("Edged image", edged)
        cv2.imshow("Lined image", line_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
