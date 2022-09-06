# Import cvzone (opencv-python must be in 4.5.5.62), mediapipe
import cv2
import numpy as np
import threading
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

frames = np.empty(2, dtype=object)
success = [False, False]


def ctlThread():
    name = "Video"
    cv2.namedWindow(name)
    while True:
        f0 = frames[0]
        f1 = frames[1]
        if not (f0 is None) and not (f1 is None):

            imgStacked = cvzone.stackImages([f0, f1], 2, 1)
            cv2.imshow(name, imgStacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting " + self.previewName)
        segmentor = SelfiSegmentation()
        camPreview(self.previewName, self.camID, segmentor)


def camPreview(previewName, camID, segmentor):
    # cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    cam.set(3, 426)  # width
    cam.set(4, 240)  # height

    while True:

        success[camID], frame = cam.read()
        frames[camID] = segmentor.removeBG(frame, (255, 0, 0), threshold=0.9)

        # if success:
        #     cv2.imshow(previewName, frameOut)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("1Failed to open " + previewName)
    cam.release()
    cv2.destroyWindow(previewName)


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
