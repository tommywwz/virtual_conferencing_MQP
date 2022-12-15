import cv2
import mediapipe as mp
import time
import numpy as np


class AutoResize:
    def __init__(self):
        self.mp_facedetector = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.ref_FIFO = []

    def resize(self, frame, ref_w, detection_confid=0.7, FIFO_len=5):

        with self.mp_facedetector.FaceDetection(min_detection_confidence=detection_confid) as face_detection:

            # Convert the BGR image to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            results = face_detection.process(frame)

            # Convert the image color back so it can be displayed
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if not results.detections:
                avg_ratio = 1
                return avg_ratio

            for result_id, detection in enumerate(results.detections):
                if DEBUG: self.mp_draw.draw_detection(frame, detection)

                bBox = detection.location_data.relative_bounding_box

                h, w, c = frame.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                # store the boundary information into a tuple with (left most, highest, right most, lowest) location
                print(boundBox)

                box_w = bBox.width*w
                print("HERE in AR: " + str(box_w))
                ref_ratio = ref_w/box_w
                if len(self.ref_FIFO) >= FIFO_len:
                    self.ref_FIFO.pop(0)

                self.ref_FIFO.append(ref_ratio)
                avg_ratio = np.average(self.ref_FIFO)

        return avg_ratio

    def bound_n_resize(self, frame, ref_w, detection_confid=0.7, FIFO_len=5):

        with self.mp_facedetector.FaceDetection(min_detection_confidence=detection_confid) as face_detection:

            # Boundary Warning flags
            bound_warn = [0, 0, 0, 0]
            # Convert the BGR image to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            results = face_detection.process(frame)

            # Convert the image color back so it can be displayed
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if not results.detections:
                avg_ratio = 1
                return avg_ratio, bound_warn

            for result_id, detection in enumerate(results.detections):
                if DEBUG:
                    self.mp_draw.draw_detection(frame, detection)
                    cv2.imshow("testb", frame)
                bBox = detection.location_data.relative_bounding_box

                h, w, c = frame.shape

                # boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                # # store the boundary information into a tuple with (left most, highest, right most, lowest) location
                # print(boundBox)

                if bBox.xmin < 0:  # left bound touched
                    bound_warn[0] = 1
                if bBox.xmin + bBox.width > 1:  # right bound touched
                    bound_warn[1] = 1
                if bBox.ymin + bBox.height > 1:  # lower bound touched
                    bound_warn[2] = 1
                if bBox.ymin < 0:  # upper bound touched
                    bound_warn[3] = 1

                print(bound_warn)  # boundary warning: [Left, Right, Down, Up], {0} if no warning

                box_w = bBox.width*w
                ref_ratio = ref_w/box_w
                if len(self.ref_FIFO) >= FIFO_len:
                    self.ref_FIFO.pop(0)

                self.ref_FIFO.append(ref_ratio)
                avg_ratio = np.average(self.ref_FIFO)

        return avg_ratio, bound_warn


DEBUG = False

if DEBUG:
    auto_rsz = AutoResize()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while cap.isOpened():
        success, raw = cap.read()

        start = time.time()

        if success:
            image = cv2.rotate(cv2.flip(raw, 1), cv2.ROTATE_90_CLOCKWISE)
            h, w, c = image.shape
            ratio, bound_warn = auto_rsz.bound_n_resize(image, 170)
            adjust_w = round(w * ratio)
            adjust_h = round(h * ratio)
            new_shape = (adjust_w, adjust_h)
            if ratio > 1:
                rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
                ah, aw = rsz_image.shape[:2]
                dn = round(ah * 0.5 + h * 0.5)
                up = round(ah * 0.5 - h * 0.5)
                lt = round(aw * 0.5 - w * 0.5)
                rt = round(aw * 0.5 + w * 0.5)
                rsz_image = rsz_image[up:dn, lt:rt]
            else:
                rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
            h, w = rsz_image.shape[:2]
            print("ratio: " + str(ratio))
            print("here" + str(rsz_image.shape))
            # Boundary Warning
            if bound_warn[0]:  # left bound touched
                cv2.line(rsz_image, (0, 0), (0, h), (0, 0, 255), 5)
            if bound_warn[1]:  # right bound touched
                cv2.line(rsz_image, (w, 0), (w, h), (0, 0, 255), 5)
            if bound_warn[2]:  # lower bound touched
                cv2.line(rsz_image, (0, h), (w, h), (0, 0, 255), 5)
            if bound_warn[3]:  # upper bound touched
                cv2.line(rsz_image, (0, 0), (w, 0), (0, 0, 255), 5)

            cv2.imshow("test", rsz_image)

        cv2.imshow("raw", raw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
