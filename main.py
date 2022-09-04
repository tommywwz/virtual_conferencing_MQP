import cv2
import numpy as np
import threading
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

if_quit = False

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)


def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    cam.set(3, 1280)  # width
    cam.set(4, 720)  # height

    while True:
        success, frame = cam.read()
        if success:
            cv2.imshow(previewName, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("1Failed to open " + previewName)
    cam.release()
    cv2.destroyWindow(previewName)


thread1 = camThread("Camera 0", 0)
thread2 = camThread("Camera 1", 1)
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
