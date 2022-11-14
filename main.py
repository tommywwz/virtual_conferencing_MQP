# Import cvzone (opencv-python must be in 4.5.5.62), mediapipe
import cv2
import numpy as np
import threading
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp
import time
import os

import edge_detection

# FRAMES = np.empty(3, dtype=object)
# TERM = False
W = 360
H = 640
SHAPE = (W, H, 3)
R = H / W  # aspect ratio using the ratio of height to width to improve the efficiency of stackIMG function
BLUE = (255, 0, 0)

vid1 = cv2.VideoCapture("vid/Single_User_View_1(WideAngle).MOV")
vid2 = cv2.VideoCapture("vid/Single_User_View_2(WideAngle).MOV")
vid3 = cv2.VideoCapture("vid/IMG_0582.MOV")
vids = [vid1, vid2, vid3]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def Yshift_img(vector, y_off, fill_clr=(0, 0, 0)):
    h, w, c = vector.shape

    blank = np.full(shape=(np.abs(y_off), w, c), fill_value=fill_clr)
    if y_off > 0:
        stack_img = np.vstack((vector, blank))
        h1, w1, c1 = stack_img.shape
        h = h1 - h
        img_out = stack_img[h:h1, 0:w, :]
    else:
        stack_img = np.vstack((blank, vector))
        img_out = stack_img[0:h, 0:w, :]

    return img_out


def stackParam(cam_list, bg_shape: int):
    # should be called everytime a new cam joined
    # return: fit_width, number of image to stack, spacing for each camera,
    bg_h, bg_w, bg_c = bg_shape
    num_of_cam = len(cam_list)
    w_step = int(np.floor(bg_w / num_of_cam))
    fit_width = w_step
    fit_height = np.floor(fit_width * R)

    if fit_height > bg_h:
        fit_width = np.floor(bg_h / R)
        fit_height = bg_h

    fit_shape = [int(fit_height), int(fit_width)]

    margin_h = np.floor((bg_h - fit_height) / 2)
    margin_w = np.floor((w_step - fit_width) / 2)
    margins = [int(margin_h), int(margin_w)]

    return fit_shape, w_step, margins


def stackIMG(cam_list, bg_img, fit_shape, w_step, margins, cam_shift_y):
    loc_bgIMG = bg_img.copy()
    fit_h, fit_w = fit_shape[0], fit_shape[1]
    h_margin, w_margin = margins[0], margins[1]
    i = 0

    for key in cam_list:
        cam = cam_list[key]
        rsz_cam = cv2.resize(cam, (fit_w, fit_h))
        rsz_cam = Yshift_img(rsz_cam, cam_shift_y[i], BLUE)
        x_left = w_step * i + w_margin
        x_right = x_left + fit_w
        y_top = h_margin
        y_bottom = h_margin + fit_h
        loc_bgIMG[y_top:y_bottom, x_left:x_right, :] = rsz_cam
        i += 1

    return loc_bgIMG


class CamManagement:
    FRAMES = {}
    TERM = False
    cam_id = 0
    edge_position = {}
    empty_frame = np.zeros(SHAPE, dtype=np.uint8)

    def open_cam(self, camID=cam_id):
        name = "Camera %s" % str(camID)
        camThread = CamThread(previewName=name, camID=camID)
        camThread.start()
        self.FRAMES[camID] = self.empty_frame
        self.edge_position[camID] = [None, None]
        self.cam_id += 1  # camera conflicts need to be fixed here
        return True

    def save_frame(self, camID, frame):
        self.FRAMES[camID] = frame

    def get_frame(self):
        return self.FRAMES.copy()

    def set_Term(self, ifTerm: bool):
        self.TERM = ifTerm

    def check_Term(self):
        return self.TERM

    def save_edge(self, camID, edge):
        self.edge_position[camID] = edge

    def get_edge(self, camID):
        return self.edge_position[camID]


class CamThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting Thread" + self.previewName)
        segmentor = SelfiSegmentation()  # setup BGremover
        camPreview(self.previewName, self.camID, segmentor)


def camPreview(previewName, camID, segmentor):
    # cam = vids[camID]

    # Real time video cap
    cam = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    ed = edge_detection.EdgeDetection()

    cam.set(3, 640)  # width
    cam.set(4, 360)  # height

    drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while cam.isOpened():
            success, frame = cam.read()

            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w, c = frame.shape

            if not success:
                # skip if no frame
                continue

            # if camID != 2:
            #     frame = cv2.resize(frame, (360, 640))
            # else:
            #     frame = cv2.resize(frame, (640, 360))
            #     bg = cv2.resize(cv2.imread("background/Bar.jpg"), (640, 360))
            #
            #     frame_bgrmv = segmentor.removeBG(frame, bg, threshold=0.5)
            #     CamMan.save_frame(camID=camID, frame=frame_bgrmv)
            #     if CamMan.check_Term():
            #         break
            #     continue

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

            edge = ed.process_frame(frame)
            a, b = edge

            if a is not None and b is not None:
                b += np.floor(h * 2 / 3)
                cv2.line(frame, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)

            frame_bgrmv = segmentor.removeBG(frame, (255, 0, 0), threshold=0.5)
            CamMan.save_frame(camID=camID, frame=frame)
            CamMan.save_edge(camID, edge)
            print("cam" + str(camID) + str(CamMan.get_edge(camID)))

            if CamMan.check_Term():
                break

        print("Exiting " + previewName)
        cam.release()


def ctlThread():
    x_off = 0
    y_off = -50
    BGdim = (640, 360)  # (w, h)

    fit_shape, w_step, margins = 0, 0, 0
    W2 = W*2
    halfW = int(W2 / 2)
    name = "Video"
    cv2.namedWindow(name)
    cv2.namedWindow("Test")
    # cv2.namedWindow("twoperson")
    imgBG = cv2.imread("background/Bar.jpg")
    imgBG = cv2.resize(imgBG, BGdim)
    cam_added = True

    CamMan.open_cam(camID=3)
    CamMan.open_cam(camID=5)
    # CamMan.open_cam()
    # thread1 = CamThread("Camera 0", 0)
    # thread2 = CamThread("Camera 1", 1)
    # # thread3 = camThread("Camera 2", 2)
    # thread1.start()
    # thread2.start()
    # # thread3.start()

    while True:
        frames = CamMan.get_frame()

        if cam_added:
            fit_shape, w_step, margins = stackParam(frames, imgBG.shape)
            cam_added = False

        cam_shift_y = [10, 0]
        imgStacked = stackIMG(frames, imgBG, fit_shape, w_step, margins, cam_shift_y)

        temp = np.subtract(imgStacked, BLUE)

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

    CamMan.set_Term(True)
    cv2.destroyWindow(name)


CamMan = CamManagement()
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
