import cv2
import threading
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os


# class camThread(threading.Thread):
#     def __init__(self, previewName, camID):
#         threading.Thread.__init__(self)
#         self.previewName = previewName
#         self.camID = camID
#
#     def run(self):
#         print("Starting " + self.previewName)
#         camPreview(self.previewName, self.camID)
#
#
# def camPreview(previewName, camID):
#     cv2.namedWindow(previewName)
#     cam = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
#     cam.set(3, 640)  # width
#     cam.set(4, 480)  # height
#     if cam.isOpened():
#         success, frame = cam.read()
#     else:
#         success = False
#         print("0Failed to open " + previewName)
#
#     while success:
#         cv2.imshow(previewName, frame)
#         success, frame = cam.read()
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     print("1Failed to open " + previewName)
#     cv2.destroyWindow(previewName)
#
#
# thread1 = camThread("Camera 0", 0)
# thread2 = camThread("Camera 1", 1)
# thread1.start()
# thread2.start()

cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1)
cap0.set(3, 640)  # width
cap0.set(4, 480)  # height
cap1.set(3, 640)  # width
cap1.set(4, 480)  # height

while True:
    ret0, img0 = cap0.read()

    if ret0:
        cv2.imshow("Image0", img0)

    ret1, img1 = cap1.read()

    if ret1:
        cv2.imshow("Image1", img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# from pymf import get_MF_devices
# device_list = get_MF_devices()
# for i, device_name in enumerate(device_list):
#     print(f"opencv_index: {i}, device_name: {device_name}")
#
# # => opencv_index: 0, device_name: Integrated Webcam
