import cv2
import numpy as np

# import pyautogui as pg

debug = test = False


# frame = cv2.imread("vid/test1.jpg")
# frame = cv2.resize(frame.copy(), (640, 480))


def gen_key(line, slope):
    [x1, y1, x2, y2] = line[0, 0:4]
    midpoint = round((y1 + y2) / 20)
    # key = slope + midpoint
    y_intercept = (y1 - (slope * x1)) / 10
    key = slope + round(y_intercept)
    return key


class EdgeDetection:
    def __init__(self):
        self.edge_height = 0
        self.stored_lines = {}
        self.lines_denoised = []
        self.counter = 0
        self.TANSLOP = np.tan(10)

    def add_lines(self, key, line):
        if self.stored_lines.get(key) is None:
            self.stored_lines[key] = [line]
        else:
            self.stored_lines[key].append(line)
        # print("counter: %d" % self.counter)
        self.counter += 1

    def filter_lines(self):
        self.counter = 0
        # max_key = max(len(item) for item in stored_lines.values())
        max_key = max(self.stored_lines, key=lambda x: len(self.stored_lines[x]))
        self.lines_denoised = self.stored_lines.get(max_key)
        self.stored_lines.clear()

    def process_frame(self, portrait_frame, threshold=70):
        if debug: cv2.imshow("raw", portrait_frame)
        h, w, c = portrait_frame.shape
        cropped_image = portrait_frame[int(np.floor(2 * h / 3)):h, :, :]
        # cropped_image = cv2.GaussianBlur(cropped_image, (11, 11), 0)
        # cropped_image = cv2.medianBlur(cropped_image, 15)
        # if debug: cv2.imshow("After first blur", cropped_image)

        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        emboss = cv2.filter2D(cropped_image, -1, kernel)
        if debug: cv2.imshow("Sharpening", emboss)

        grey = cv2.cvtColor(emboss, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (25, 1), 0)
        if debug: cv2.imshow("After Gaussian blur", blurred)
        blurred = cv2.medianBlur(blurred, 9)
        if debug: cv2.imshow("After median blur", blurred)

        edged = cv2.Canny(blurred, threshold1=10, threshold2=50)
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180,
                                threshold=15, lines=np.array([]), minLineLength=30, maxLineGap=3)
        line_image = cropped_image.copy()

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    dy = y1 - y2
                    dx = x1 - x2
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if np.abs(dy) < np.abs(dx) * self.TANSLOP:
                        slope = round(dy / dx, 1)
                        key = gen_key(line, slope)

                        self.add_lines(key=key, line=line)
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if self.counter > threshold:
            self.filter_lines()

        if self.lines_denoised:
            selected_lines = self.lines_denoised
            x_values = []
            y_values = []
            for denoised_line in selected_lines:
                for x1, y1, x2, y2 in denoised_line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # selected candidates
                    x_values.extend([x1, x2])
                    y_values.extend([y1, y2])

            # calculate linear regression of selected samples
            A = np.vstack([x_values, np.ones(len(x_values))]).T
            y_values = np.array(y_values)[:, np.newaxis]
            alpha = np.linalg.lstsq(A, y_values, rcond=None)[0]  # find linear least square regression
            # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter16.04-Least-Squares-Regression-in-Python.html
            self.edge_height = np.mean(y_values)
            w = line_image.shape[1]
            a = alpha[0, 0]
            b = alpha[1, 0]
            cv2.line(line_image, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)
            if debug:
                cv2.imshow("after processing", line_image)
            b += np.floor(h * 2 / 3)  # convert to full frame coordinate
            retval = [a, b]

        else:
            retval = [None, None]

        # print(retval)
        return retval
    # cv2.imshow("lined image", line_image)


if test:
    from pymf import get_MF_devices
    device_list = get_MF_devices()
    for i, device_name in enumerate(device_list):
        print(f"opencv_index: {i}, device_name: {device_name}")

    # => opencv_index: 0, device_name: Integrated Webcam

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 360)
    ed = EdgeDetection()

    while cam.isOpened():
        success, raw_frame = cam.read()

        if success:
            cv2.imshow("original", raw_frame)
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ed.process_frame(raw_frame, 100)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.release()
                break
