import cv2
import mediapipe as mp
import time
import numpy as np


class AutoResize:
    def __init__(self):
        self.mp_facedetector = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.ref_FIFO = []

    def bound_n_resize(self, frame, ref_w, detection_confid=0.7, FIFO_len = 5):

        with self.mp_facedetector.FaceDetection(min_detection_confidence=detection_confid) as face_detection:

            # Boundary Warning flags
            bound_warn = [0, 0, 0, 0]
            # Convert the BGR image to RGB
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Process the image and find faces
            results = face_detection.process(frame)

            # Convert the image color back so it can be displayed
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.detections:
                for id, detection in enumerate(results.detections):
                    if DEBUG: self.mp_draw.draw_detection(frame, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    # store the boundary information into a tuple with (left most, highest, right most, lowest) location
                    print(boundBox)

                    # Boundary Warning
                    if bBox.xmin < 0:  # left bound touched
                        bound_warn[0] = 1
                        cv2.line(frame, (0, 0), (0, h), (0, 0, 255), 5)
                    if bBox.xmin+bBox.width > 1:  # right bound touched
                        bound_warn[1] = 1
                        cv2.line(frame, (w, 0), (w, h), (0, 0, 255), 5)
                    if bBox.ymin+bBox.height > 1:  # lower bound touched
                        bound_warn[2] = 1
                        cv2.line(frame, (0, h), (w, h), (0, 0, 255), 5)
                    if bBox.ymin < 0:  # upper bound touched
                        bound_warn[3] = 1
                        cv2.line(frame, (0, 0), (w, 0), (0, 0, 255), 5)

                    print(bound_warn)  # boundary warning: [Left, Right, Down, Up], {0} if no warning

                    box_w = bBox.width*w
                    ref_ratio = ref_w/box_w
                    if len(self.ref_FIFO) >= FIFO_len:
                        self.ref_FIFO.pop(0)

                    self.ref_FIFO.append(ref_ratio)
                    avg_ratio = np.average(self.ref_FIFO)

                    adjust_w = round(w*avg_ratio)
                    adjust_h = round(h*avg_ratio)

                    print("here" + str(adjust_w))
                    frame = cv2.resize(frame, (adjust_w, adjust_h))
                    # if (boundBox[3] - reference) > 0.15*reference:
                    #     w = round(0.9*w)
                    #     h = round(0.9*h)
                    #     frame = cv2.resize(frame, (w, h))
                    # if (reference - boundBox[3]) > 0.15*reference:
                    #     w = round(1.1 * w)
                    #     h = round(1.1 * h)
                    #     frame = cv2.resize(frame, (w, h))

        return frame

DEBUG = True
if DEBUG:
    auto_rsz = AutoResize()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while cap.isOpened():
        success, image = cap.read()

        start = time.time()

        if success:
            output = auto_rsz.bound_n_resize(image, 100)
            raw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("raw", raw)
        cv2.imshow("test", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
