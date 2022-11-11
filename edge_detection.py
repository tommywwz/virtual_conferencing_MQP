import cv2
import numpy as np
import matplotlib.pyplot as plt

# import pyautogui as pg


cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(3, 640)  # width
cam.set(4, 360)  # height


# frame = cv2.imread("vid/test1.jpg")
# frame = cv2.resize(frame.copy(), (640, 480))


# stored_lines = {}
# lines_denoised = []
# counter = 0


class EdgeCollection:
    stored_lines = {}
    lines_denoised = []
    counter = 0
    TANSLOP = np.tan(10)

    def add_lines(self, key, line):
        if self.stored_lines.get(key) is None:
            self.stored_lines[key] = [line]
        else:
            self.stored_lines[key].append(line)
        print(self.stored_lines)
        self.counter += 1

    def filter_lines(self):
        self.counter = 0
        # max_key = max(len(item) for item in stored_lines.values())
        max_key = max(self.stored_lines, key=lambda x: len(self.stored_lines[x]))
        self.lines_denoised = self.stored_lines.get(max_key)
        self.stored_lines.clear()


def gen_key(line, slope):
    [x1, y1, x2, y2] = line[0, 0:4]
    midpoint = round((y1 + y2) / 20)
    # key = slope + midpoint
    y_intercept = (y1-(slope * x1))/10
    key = slope + round(y_intercept)
    return key


def process_frame(portrait_frame):
    cv2.imshow("raw", portrait_frame)
    h, w, c = portrait_frame.shape
    cropped_image = portrait_frame[int(np.floor(2 * h / 3)):h, :, :]
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    im = cv2.filter2D(cropped_image, -1, kernel)
    cv2.imshow("Sharpening", im)
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (21, 7), 0)
    edged = cv2.Canny(blurred, threshold1=10, threshold2=50)
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180,
                            threshold=15, lines=np.array([]), minLineLength=30, maxLineGap=3)
    line_image = cropped_image.copy()

    if edge_coll.counter > 70:
        debug_alllines = edge_coll.stored_lines
        edge_coll.filter_lines()

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                dy = y1 - y2
                dx = x1 - x2
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if np.abs(dy) < np.abs(dx) * edge_coll.TANSLOP:
                    slope = round(dy/dx, 1)
                    key = gen_key(line, slope)

                    edge_coll.add_lines(key=key, line=line)
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if edge_coll.lines_denoised:
        for line_denoised in edge_coll.lines_denoised:
            for x1, y1, x2, y2 in line_denoised:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # selected candidates

    cv2.imshow("lined image", line_image)


edge_coll = EdgeCollection()

while cam.isOpened():
    success, raw_frame = cam.read()

    if success:
        frame = cv2.rotate(raw_frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        process_frame(frame)
        # while rawimage is not None:
        #     success = True

        # if success:
        #     rawimage = cv2.rotate(rawimage, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     cv2.imshow("raw", rawimage)
        #     h, w, c = rawimage.shape
        #     cropped_image = rawimage[int(np.floor(2 * h / 3)):h, :, :]
        #     kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        #     im = cv2.filter2D(cropped_image, -1, kernel)
        #     cv2.imshow("Sharpening", im)
        #     grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #     blurred = cv2.GaussianBlur(grey, (21, 7), 0)
        #     edged = cv2.Canny(blurred, threshold1=10, threshold2=50)
        #     lines = cv2.HoughLinesP(edged, 1, np.pi / 180,
        #                             threshold=15, lines=np.array([]), minLineLength=30, maxLineGap=3)
        #     # stored_lines = lines
        #     line_image = cropped_image.copy()
        #
        #     if counter > 70:
        #         counter = 0
        #         # max_key = max(len(item) for item in stored_lines.values())
        #         max_key = max(stored_lines, key=lambda x: len(stored_lines[x]))
        #         lines_denoised = stored_lines.get(max_key)
        #         stored_lines.clear()
        #
        #     if lines is not None:
        #         # print(lines)
        #         # stored_lines.clear()
        #         for line in lines:
        #             for x1, y1, x2, y2 in line:
        #                 dy = y1 - y2
        #                 dx = x1 - x2
        #                 cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #                 if np.abs(dy) < np.abs(dx) * TANSLOP:
        #                     cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        #                     slope = round(dy / dx, 3)
        #                     midpoint = round((y1 + y2) / 20)
        #                     key = slope + midpoint
        #                     if stored_lines.get(key) is None:
        #                         stored_lines[key] = [line]
        #                     else:
        #                         stored_lines[key].append(line)
        #                     print(stored_lines)
        #                     counter += 1
        #
        #     if lines_denoised:
        #         for line_denoised in lines_denoised:
        #             for x1, y1, x2, y2 in line_denoised:
        #                 cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #
        #     # elif stored_lines:
        #     #     print(stored_lines)
        #     #     for lines in stored_lines.values():
        #     #         for line in lines:
        #     #             for x1, y1, x2, y2 in line:
        #     #                 cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        #
        #     # line_image = cv2.addWeighted(image, 1, line_image, 1, 0)
        #     cv2.imshow("Original image", cropped_image)
        #     cv2.imshow("Blured image", blurred)
        #     cv2.imshow("Edged image", edged)
        #     cv2.imshow("Lined image", line_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
