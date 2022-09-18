# Import cvzone (opencv-python must be in 4.5.5.62), mediapipe
import cv2
import numpy as np
import threading
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp
import time
import os

FRAMES = np.empty(3, dtype=object)
TERM = False

vid1 = cv2.VideoCapture("vid/Single_User_View_1(WideAngle).MOV")
vid2 = cv2.VideoCapture("vid/Single_User_View_2(WideAngle).MOV")
vid3 = cv2.VideoCapture("vid/IMG_0582.MOV")
vids = [vid1, vid2, vid3]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def ctlThread():
    H = 640
    W = 720
    name = "Video"
    cv2.namedWindow(name)
    cv2.namedWindow("Test")
    cv2.namedWindow("twoperson")
    imgBG = cv2.imread("background/Bar.jpg")
    imgBG = cv2.resize(imgBG, (W, H))

    thread1 = camThread("Camera 0", 0)
    thread2 = camThread("Camera 1", 1)
    thread3 = camThread("Camera 2", 2)
    thread1.start()
    thread2.start()
    thread3.start()

    while True:
        global FRAMES
        f0 = FRAMES[0]
        f1 = FRAMES[1]
        f2 = FRAMES[2]

        if f2 is not None:
            cv2.imshow("twoperson", f2)

        if f0 is not None and f1 is not None:

            # print(f0.shape)
            imgStacked = cvzone.stackImages([f0, f1], 2, 1)
            # print(imgStacked.shape)
            # imgStackedGBRA = cv2.cvtColor(imgStacked, cv2.COLOR_BGR2BGRA)
            color = (255, 0, 0)  # color to make transparent
            temp = np.subtract(imgStacked, color)

            # Transparent mask stores boolean value
            mask = (temp == (0, 0, 0))
            mask_singleCH = (mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2])

            alpha = np.zeros(imgStacked.shape, dtype=np.uint8)
            imgBG_output = imgBG.copy()
            # imgBG_output[mask_bin, :] = imgBG[mask_bin, :]
            imgBG_output[~mask_singleCH, :] = imgStacked[~mask_singleCH, :]

            alpha[mask_singleCH] = 0
            alpha[~mask_singleCH] = 255
            alpha_grey = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)

            contours, hierarchy = cv2.findContours(alpha_grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_copy = imgBG_output.copy()
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                             lineType=cv2.LINE_AA)
            blurred_img = cv2.GaussianBlur(imgBG_output, (9, 9), 0)
            output = np.where(mask == np.array([0, 255, 0]), blurred_img, imgBG_output)
            # BG = imgBG.copy()
            # BG[0:640, 0:720, :] = cv2.cvtColor(imgStackedGBRA, cv2.COLOR_BGRA2BGR)
            # imgDisplay = imgBG.copy()
            # imgDisplay [0:848, 0:480, :] = f0[0:848, 0:480, :]

            cv2.imshow(name, imgBG_output)
            cv2.imshow("Test", alpha_grey)

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
    # cv2.flip()
    # cam.set(3, 640)  # width
    # cam.set(4, 360)  # height

    drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                # print("empty frame!")
                continue

            if camID != 2:
                frame = cv2.resize(frame, (360, 640))
            else:
                frame = cv2.resize(frame, (640, 360))
                bg = cv2.resize(cv2.imread("background/Bar.jpg"), (640, 360))

                FRAMES[camID] = segmentor.removeBG(frame, bg, threshold=0.5)
                continue

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

            FRAMES[camID] = segmentor.removeBG(frame, (255, 0, 0), threshold=0.5)

            # if success:
            #     cv2.imshow(previewName, frameOut)
            if TERM:
                break
        print("Exiting " + previewName)
        cam.release()


thread0 = threading.Thread(target=ctlThread)
thread0.start()


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
