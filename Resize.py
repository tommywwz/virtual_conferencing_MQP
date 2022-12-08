import cv2
import mediapipe as mp
import time
import numpy as np
from scipy.ndimage import zoom


mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)


# not using function below
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():

        success, image = cap.read()

        start = time.time()
        bondWarn = [0, 0, 0, 0]
        # Convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Process the image and find faces
        results = face_detection.process(image)

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(image, detection)
                #print(id, detection)

                bBox = detection.location_data.relative_bounding_box

                h, w, c = image.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                print(boundBox)
                if boundBox[0] < 20:
                    bondWarn[0] = 1
                    cv2.line(image, (0, 0), (0, 640), (0, 0, 255), 5)
                if boundBox[0] > 340:
                    bondWarn[1] = 1
                    cv2.line(image, (480, 0), (480, 640), (0, 0, 255), 5)
                if boundBox[1] < 300:
                    bondWarn[2] = 1
                    cv2.line(image, (0, 0), (480, 0), (0, 0, 255), 5)
                if boundBox[1] > 400:
                    bondWarn[3] = 1
                    cv2.line(image, (0, 640), (480, 640), (0, 0, 255), 5)

                print(bondWarn)  # boundary warning: [Left, Right, Up, Down], {0} if no warning

                # cv2.putText(image, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        #if boundBox[2] < 100 & abs(boundBox[2] - 100) <= 20:
            #image = clipped_zoom(image, 1.2)



        end = time.time()
        totalTime = end - start

        # fps = 1 / totalTime
        # print("FPS: ", fps)
        #
        # cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow('Face Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()