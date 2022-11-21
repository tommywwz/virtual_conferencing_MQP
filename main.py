# Import cvzone (opencv-python must be in 4.5.5.62), mediapipe
import cv2
import numpy as np
import threading
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp
import time
import logging
import os

import edge_detection

logging.basicConfig(level=logging.DEBUG)

BG_W = 640
BG_H = 360
BG_DIM = (BG_W, BG_H)  # (w, h)
CAM_W = 360
CAM_H = 640
SHAPE = (CAM_W, CAM_H, 3)
R = CAM_H / CAM_W  # aspect ratio using the ratio of height to width to improve the efficiency of stackIMG function
BLUE = (255, 0, 0)

vid1 = cv2.VideoCapture("vid/Single_User_View_1(WideAngle).MOV")
vid2 = cv2.VideoCapture("vid/Single_User_View_2(WideAngle).MOV")
vid3 = cv2.VideoCapture("vid/IMG_0582.MOV")
vids = [vid1, vid2, vid3]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def Yshift_img(img_vector, y_offset: int, fill_clr=(0, 0, 0)):
    # y_offset: positive: upward shifting; negative: downward shifting

    h, w, c = img_vector.shape

    blank = np.full(shape=(np.abs(y_offset), w, c), fill_value=fill_clr)
    if y_offset > 0:
        stack_img = np.vstack((img_vector, blank))
        h1, w1, c1 = stack_img.shape
        h = h1 - h
        img_out = stack_img[h:h1, 0:w, :]
    else:
        stack_img = np.vstack((blank, img_vector))
        img_out = stack_img[0:h, 0:w, :]

    return img_out


def stackParam(cam_dict, bg_shape: int):
    # should be called everytime a new cam joined
    # return: fit_width, number of image to stack, spacing for each camera,
    bg_h, bg_w, bg_c = bg_shape
    num_of_cam = len(cam_dict)
    w_step = int(np.floor(bg_w / num_of_cam))
    fit_width = w_step
    fit_height = np.floor(fit_width * R)

    if fit_height > bg_h:
        fit_width = np.floor(bg_h / R)
        fit_height = bg_h

    fit_shape = (int(fit_height), int(fit_width))

    margin_h = np.floor((bg_h - fit_height) / 2)
    margin_w = np.floor((w_step - fit_width) / 2)
    margins = [int(margin_h), int(margin_w)]

    return fit_shape, w_step, margins


def stackIMG(cam_dict, bg_img, fit_shape, w_step, margins, cam_shift_y):
    loc_bgIMG = bg_img.copy()
    fit_h, fit_w = fit_shape
    h_margin, w_margin = margins[0], margins[1]
    i = 0

    for camID in cam_dict:
        frame = cam_dict[camID]
        rsz_cam = cv2.resize(frame, (fit_w, fit_h))
        rsz_cam = Yshift_img(rsz_cam, cam_shift_y[camID], BLUE)
        x_left = w_step * i + w_margin
        x_right = x_left + fit_w
        y_top = h_margin
        y_bottom = h_margin + fit_h
        loc_bgIMG[y_top:y_bottom, x_left:x_right, :] = rsz_cam
        i += 1

    return loc_bgIMG


class CamManagement:
    cam_id = 0
    reference_y = np.floor(BG_DIM[1]*6/7)

    def __init__(self):
        self.FRAMES = {}
        self.TERM = False
        self.edge_lines = {}  # edge equation (a, b) for each cam
        self.edge_y = {}  # average height of edge for each cam
        self.empty_frame = np.zeros(SHAPE, dtype=np.uint8)
        self.calib = True
        self.calibCam = None

    def open_cam(self, camID=cam_id, if_user=False):
        cam_name = "Camera %s" % str(camID)
        camThread = CamThread(cam_name, camID, if_user)
        camThread.start()
        time.sleep(0.5)
        logging.info("%s: starting", cam_name)
        self.FRAMES[camID] = self.empty_frame
        self.edge_lines[camID] = [None, None]
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
        a, b = edge
        if a is not None and b is not None:
            self.edge_lines[camID] = edge
            self.edge_y[camID] = int(np.floor(CAM_W * a / 2 + b))

    def get_edge(self):
        return self.edge_lines

    def get_cam_offset(self):
        offset = {}
        for camID in self.FRAMES:
            if camID in self.edge_y.keys():
                background_y = np.floor(self.edge_y[camID]*BG_H/CAM_H)
                offset[camID] = int(background_y - self.reference_y)
            else:
                offset[camID] = 0
        return offset

    def toggle_calib(self, camID):
        self.calib = not self.calib


class CamThread(threading.Thread):
    def __init__(self, previewName, camID, if_usercam):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.if_usercam = if_usercam

    def run(self):
        print("Starting Thread" + self.previewName)
        segmentor = SelfiSegmentation()  # setup BGremover
        camPreview(self.previewName, self.camID, segmentor, self.if_usercam)


def camPreview(previewName, camID, segmentor, if_usercam):
    # cam = vids[camID]
    # cv2.namedWindow("iso frame " + str(camID))
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

            if CamMan.calib and if_usercam:  # check if calibration is toggled in the user's cam thread
                edge = ed.process_frame(frame, threshold=100)
                a, b = edge
                CamMan.save_edge(camID, [a, b])
                if a is not None and b is not None:
                    cv2.line(frame, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)

            frame_bgrmv = segmentor.removeBG(frame, (255, 0, 0), threshold=0.5)
            CamMan.save_frame(camID=camID, frame=frame)
            logging.debug(str(CamMan.get_edge()))

            if CamMan.check_Term():
                break

        print("Exiting " + previewName)
        cam.release()


def ctlThread():
    userCam = 1
    clientCam = 0
    fit_shape, w_step, margins = 0, 0, 0
    W2 = CAM_W * 2
    halfW = int(W2 / 2)
    cam_loaded = 0

    name = "Video"
    cv2.namedWindow(name)

    imgBG = cv2.imread("background/Bar.jpg")
    imgBG = cv2.resize(imgBG, BG_DIM)

    CamMan.open_cam(camID=userCam, if_user=True)
    CamMan.open_cam(camID=clientCam)
    # CamMan.open_cam()

    while True:
        frame_dict = CamMan.get_frame()
        if not frame_dict:
            continue  # if frame dictionary is empty, continue

        if CamMan.calib:  # if calibration is toggled by user
            cv2.imshow("calibration window", frame_dict[userCam])
            frame_dict.pop(userCam, None)

        cam_count = len(frame_dict.keys())
        if cam_count != cam_loaded:  # update the stack parameter everytime a new cam joined
            cam_loaded = len(frame_dict.keys())
            fit_shape, w_step, margins = stackParam(frame_dict, imgBG.shape)

        cam_offset_y = CamMan.get_cam_offset()

        if cam_offset_y:
            imgStacked = stackIMG(frame_dict, imgBG, fit_shape, w_step, margins, cam_offset_y)
        else:
            for cam in frame_dict:
                cam_offset_y[cam] = 0
            imgStacked = stackIMG(frame_dict, imgBG, fit_shape, w_step, margins, cam_offset_y)

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

        # cv2.imshow("Test", alpha_grey)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('t'):
            CamMan.toggle_calib(userCam)

    CamMan.set_Term(True)
    cv2.destroyWindow(name)


CamMan = CamManagement()
logging.info("Starting Control Thread")
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
