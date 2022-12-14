import cv2
import mediapipe as mp
import time
import numpy as np

DEBUG = True


def bound_n_resize(frame, reference):
    import mediapipe as mp
    mp_facedetector = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils

    with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

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
                mp_draw.draw_detection(frame, detection)
                #print(id, detection)

                bBox = detection.location_data.relative_bounding_box

                h, w, c = frame.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                print(boundBox)

                # Boundary Warning
                if boundBox[0] < 20:
                    bound_warn[0] = 1
                    cv2.line(frame, (0, 0), (0, 640), (0, 0, 255), 5)
                if boundBox[0] > 340:
                    bound_warn[1] = 1
                    cv2.line(frame, (480, 0), (480, 640), (0, 0, 255), 5)
                if boundBox[1] < 300:
                    bound_warn[2] = 1
                    cv2.line(frame, (0, 0), (480, 0), (0, 0, 255), 5)
                if boundBox[1] > 400:
                    bound_warn[3] = 1
                    cv2.line(frame, (0, 640), (480, 640), (0, 0, 255), 5)

                print(bound_warn)  # boundary warning: [Left, Right, Up, Down], {0} if no warning

                if (boundBox[3] - reference) > 0.15*reference:
                    w = round(0.9*w)
                    h = round(0.9*h)
                    frame = cv2.resize(frame, (w, h))
                if (reference - boundBox[3]) > 0.15*reference:
                    w = round(1.1 * w)
                    h = round(1.1 * h)
                    frame = cv2.resize(frame, (w, h))

    return frame









if DEBUG:
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while cap.isOpened():
        success, image = cap.read()

        start = time.time()

        if success:
            output = bound_n_resize(image, 120)
            raw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


        end = time.time()
        totalTime = end - start

        cv2.imshow("raw", raw)
        cv2.imshow("test", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
