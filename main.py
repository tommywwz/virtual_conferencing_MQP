# Import cvzone (opencv-python must be in 4.5.5.62), mediapipe
import cv2
import numpy as np
import threading
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp
import time
import os

FRAMES = np.empty(2, dtype=object)
TERM = False

vid1 = cv2.VideoCapture("vid/Single_User_View_1(WideAngle).MOV")
vid2 = cv2.VideoCapture("vid/Single_User_View_2(WideAngle).MOV")
vids = [vid1, vid2]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def ctlThread():
    name = "Video"
    cv2.namedWindow(name)
    imgBG = cv2.imread("background/Bar.jpg")
    imgBG = cv2.resize(imgBG, (848, 480))

    while True:
        f0 = FRAMES[0]
        f1 = FRAMES[1]
        if f0 is not None and f1 is not None:

            imgStacked = cvzone.stackImages([f0, f1], 2, 1)
            cv2.imshow(name, imgStacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    global TERM
    TERM = True
    cv2.destroyWindow(name)


class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting " + self.previewName)
        segmentor = SelfiSegmentation() #setup BGremover
        camPreview(self.previewName, self.camID, segmentor)


def camPreview(previewName, camID, segmentor):
    # cv2.namedWindow(previewName)

    # Real time video cap
    # cam = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    cam = vids[camID]

    # cam.set(3, 640)  # width
    # cam.set(4, 360)  # height

    drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                # print("empty frame!")
                continue

            frame = cv2.resize(frame, (480, 848))

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:

                for face_landmarks in results.multi_face_landmarks:

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

            FRAMES[camID] = segmentor.removeBG(frame, threshold=0.6)

            # if success:
            #     cv2.imshow(previewName, frameOut)
            if TERM:
                break
        print("Exiting " + previewName)
        cam.release()


thread0 = threading.Thread(target=ctlThread)
thread1 = camThread("Camera 0", 0)
thread2 = camThread("Camera 1", 1)
thread0.start()
thread1.start()
thread2.start()

# cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap1 = cv2.VideoCapture(1)
# cap0.set(3, 640)  # width
# cap0.set(4, 480)  # height
# cap1.set(3, 640)  # width
# cap1.set(4, 480)  # height
#
# while True:
#     ret0, img0 = cap0.read()
#
#     if ret0:
#         cv2.imshow("Image0", img0)
#     ret1, img1 = cap1.read()
#     if ret1:
#         cv2.imshow("Image1", img1)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# from pymf import get_MF_devices
# device_list = get_MF_devices()
# for i, device_name in enumerate(device_list):
#     print(f"opencv_index: {i}, device_name: {device_name}")
#
# # => opencv_index: 0, device_name: Integrated Webcam
