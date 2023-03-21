import cv2
import numpy as np


# import pyautogui as pg
# frame = cv2.imread("vid/test1.jpg")
# frame = cv2.resize(frame.copy(), (640, 480))
MAX_SLOP = 10


def gen_key(line, slope):
    # generate kep for categorise the lines
    [x1, y1, x2, y2] = line[0, 0:4]
    y = y_intersection(slope, (x1, y1))
    key = slope + round(y)
    return key


def y_intersection(slope, point, scale_factor=0.1):
    # a function to calculate the y intersection with a slope and a point
    x, y = point
    return (y - (slope * x)) * scale_factor


def get_regress(list_of_lines):
    x_values = []
    y_values = []
    for denoised_line in list_of_lines:
        for x1, y1, x2, y2 in denoised_line:
            x_values.extend([x1, x2])
            y_values.extend([y1, y2])

    # calculate linear regression of selected samples
    x_vals = np.vstack([x_values, np.ones(len(x_values))]).T
    y_vals = np.array(y_values)[:, np.newaxis]
    alpha = np.linalg.lstsq(x_vals, y_vals, rcond=None)[0]  # find linear least square regression
    # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter16.04-Least-Squares-Regression-in-Python.html
    edge_height = np.mean(y_vals)
    a = alpha[0, 0]
    b = alpha[1, 0]
    return a, b


class EdgeLines:
    def __init__(self, line, init_length):
        self.myLength = init_length  # the length of the candidates list
        self.lines = [line]  # a list of lines (x1, x2, y1, y2)
        self.a_b = None  # the slope and y intersection (Tuple)

    def appendLine(self, line):
        self.myLength += 1
        self.lines.append(line)

    def evaluateLine(self):  # calculate linear regression of selected samples
        self.a_b = get_regress(self.lines)
        return self.a_b


class EdgeDetection:
    def __init__(self):
        self.edge_height = 0
        self.line_candidates = {}  # a dictionary of EdgeLine data structure
        self.sample_counter = 0
        self.TANSLOP = np.tan(MAX_SLOP)
        self.regress_line = None
        self.y_offset = 0  # records the offset of y coordinate after the raw picture is cropped

    def add_line(self, key, line):
        if self.line_candidates.get(key) is None:
            edgeLines = EdgeLines(line, init_length=1)
            self.line_candidates[key] = edgeLines
        else:
            self.line_candidates[key].appendLine(line)
        # print("counter: %d" % self.counter)
        self.sample_counter += 1

    def get_relevance(self, a, b, prefer_point):
        # get the relevance value between the given point and user preferred point
        if prefer_point is None:  # if user haven't selected a point
            return 0
        prefer_x, prefer_y = prefer_point
        offset_PP = (prefer_x, prefer_y - self.y_offset)
        prefer_y = y_intersection(a, offset_PP, 1)
        actual_y = b
        relevance = np.abs(prefer_y - actual_y)
        print("relevance: ", relevance)
        return relevance

    def filter_lines(self, prefer_point):
        new_cand_dict = {}

        self.sample_counter = 0
        # max_key = max(len(item) for item in stored_lines.values())

        for key in self.line_candidates:
            edgeLine_obj = self.line_candidates[key]
            list_length = edgeLine_obj.myLength
            if list_length > 1:  # filter out the single item list
                a, b = edgeLine_obj.evaluateLine()
                edgeLine_obj.relevance = self.get_relevance(a, b, prefer_point)

                # The new key is generated base on the number of lines in the group
                # and how close (the relevance) this group to user preference
                new_key = list_length - edgeLine_obj.relevance
                new_cand_dict[new_key] = edgeLine_obj

        biggest_key = max(new_cand_dict.keys())
        # max_key = max(self.line_candidates, key=lambda x: len(self.line_candidates[x]))
        best_edgeLine_obj = new_cand_dict[biggest_key]
        self.line_candidates.clear()
        return best_edgeLine_obj

    def process_frame(self, portrait_frame, sample_size=70, prefer_point=None):
        # if debug: cv2.imshow("raw", portrait_frame)
        h, w, c = portrait_frame.shape
        self.y_offset = np.floor(h * 2 / 3)
        cropped_image = portrait_frame[int(np.floor(2 * h / 3)):h, :, :]
        cropped_image = cv2.medianBlur(cropped_image, 7)

        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        emboss = cv2.filter2D(cropped_image, -1, kernel)
        # if debug: cv2.imshow("Sharpening", emboss)

        grey = cv2.cvtColor(emboss, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (25, 3), 0)
        # if debug: cv2.imshow("After Gaussian blur", blurred)

        edged = cv2.Canny(blurred, threshold1=10, threshold2=50)
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180,
                                threshold=15, lines=np.array([]), minLineLength=30, maxLineGap=3)
        line_image = cropped_image.copy()

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    dy = y1 - y2
                    dx = x1 - x2
                    # cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if np.abs(dy) < np.abs(dx) * self.TANSLOP:
                        slope = round(dy / dx, 1)
                        # relevance = get_relevance((x1, y1), prefer_point, prefer_point)
                        key = gen_key(line, slope)
                        self.add_line(key=key, line=line)
                        # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if self.sample_counter > sample_size:  # when collected enough samples
            filted_edgeLine_obj = self.filter_lines(prefer_point)

            with filted_edgeLine_obj:
                a, b = filted_edgeLine_obj.a_b

                cv2.line(line_image, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)
                # if debug:
                #     cv2.imshow("after processing", line_image)
                b += self.y_offset  # convert to full frame coordinate
                self.regress_line = [a, b]

        # print(self.regress_line)
        return self.regress_line
# cv2.imshow("lined image", line_image)


debug = test = False

if test:
    from pymf import get_MF_devices

    device_list = get_MF_devices()
    for i, device_name in enumerate(device_list):
        print(f"opencv_index: {i}, device_name: {device_name}")

    # => opencv_index: 0, device_name: Integrated Webcam

    # cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cam.set(3, 640)
    # cam.set(4, 360)
    cam = cv2.VideoCapture("vid/demo6.mp4")
    ed = EdgeDetection()

    while cam.isOpened():
        success, raw_frame = cam.read()

        if success:
            cv2.imshow("original", raw_frame)
            # raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ed.process_frame(raw_frame, 100)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.release()
                break
