import os
import time

import cv2
import numpy as np
import threading
import logging
from queue import Queue
from Utils import edge_detection, HeadTrack, Params, CamManagement
from Utils import bg_remove_mp as bgmp
from Utils.AutoResize import AutoResize
from Utils.Frame import Frame
from server import video_server

SERVER_CAM_ID = 0

logging.basicConfig(level=logging.DEBUG)
FRAMES_lock = threading.Lock()

current_path = os.path.dirname(__file__)
root_path = os.path.split(current_path)[0]

vid1 = root_path + "/assets/vid/demo2.mp4"
vid2 = root_path + "/assets/vid/demo1.mp4"
vid3 = root_path + "/assets/vid/demo3.mp4"
vid4 = root_path + "/assets/vid/demo4.mp4"
vid5 = root_path + "/assets/vid/demo5.mp4"
vid6 = root_path + "/assets/vid/demo6.mp4"
vids = {101: vid1, 102: vid2, 103: vid3, 104: vid4, 105: vid5, 106: vid6}


cam_mutex = threading.Lock()


def do_calib(edge_detector, frame, mouse_location):
    font = cv2.FONT_HERSHEY_SIMPLEX
    linetype = cv2.LINE_AA

    edge = edge_detector.process_frame(frame, sample_size=100, prefer_point=mouse_location)
    cv2.circle(frame, mouse_location, radius=3, color=(255, 0, 255), thickness=-1)
    cv2.putText(frame,
                text='Please make sure your hands are below the table',
                org=(10, Params.VID_H - 10), fontFace=font, fontScale=.4, color=(0, 0, 255),
                thickness=1, lineType=linetype, bottomLeftOrigin=False)

    if edge is None:  # invalid edge line
        cv2.putText(frame,
                    text='Edge is not detected!',
                    org=(10, Params.VID_H - 30), fontFace=font, fontScale=.4, color=(0, 0, 255),
                    thickness=1, lineType=linetype, bottomLeftOrigin=False)
        return frame, edge

    a, b = edge
    h, w, c = frame.shape
    left_intercept = b
    right_intercept = w * a + b
    if left_intercept > h or a > 0.15:
        # check if the edge is intercept with the bottom line of screen
        cv2.putText(frame,
                    text='Try to rotate your camera to the left',
                    org=(10, Params.VID_H - 30), fontFace=font, fontScale=.4, color=(0, 255, 255),
                    thickness=1, lineType=linetype, bottomLeftOrigin=False)
        cv2.line(frame, (0, round(left_intercept)), (w, round(right_intercept)), (0, 255, 255), 2)
    elif right_intercept > h or a < -0.15:
        cv2.putText(frame,
                    text='Try to rotate your camera to the right',
                    org=(10, Params.VID_H - 30), fontFace=font, fontScale=.4, color=(0, 255, 255),
                    thickness=1, lineType=linetype, bottomLeftOrigin=False)
        cv2.line(frame, (0, round(left_intercept)), (w, round(right_intercept)), (0, 255, 255), 2)
    else:
        cv2.putText(frame,
                    text='Perfect!',
                    org=(10, Params.VID_H - 30), fontFace=font, fontScale=.4, color=(0, 255, 0),
                    thickness=1, lineType=linetype, bottomLeftOrigin=False)
        cv2.line(frame, (0, round(left_intercept)), (w, round(right_intercept)), (0, 255, 0), 2)

    return frame, edge


class CamThread(threading.Thread):
    def __init__(self, previewName, camID, if_usercam, if_demo):
        threading.Thread.__init__(self)
        self.CamMan_singleton = CamManagement.CamManagement()
        self.previewName = previewName
        self.camID = camID
        self.user_cam_id = 0
        self.user_cam = None
        self.if_usercam = if_usercam
        self.if_demo = if_demo
        self.mouse_location = None

    def run(self):
        print("Starting Thread: " + self.previewName)
        if self.if_demo:  # if starting a demo video, starts videoPreview, otherwise, starts camera thread
            self.videoPreview(self.previewName, self.camID)
        else:
            self.camPreview(self.previewName, self.user_cam_id)

    def set_user_cam(self, cam_id):
        with cam_mutex:
            if self.user_cam is not None and self.user_cam_id == cam_id:
                return
            self.user_cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            self.user_cam.set(3, Params.RAW_CAM_W)  # width
            self.user_cam.set(4, Params.RAW_CAM_H)  # height
            self.user_cam_id = cam_id

    def camPreview(self, previewName, camID):
        # cam = vids[camID]
        # cv2.namedWindow("iso frame " + str(camID))
        # Real time video cap
        self.set_user_cam(camID)
        edge_detector = edge_detection.EdgeDetection()
        empty_frame = np.zeros((Params.VID_H, Params.VID_W, 3), np.uint8)

        while True:
            with cam_mutex:
                success, frame = self.user_cam.read()
                isOpened = self.user_cam.isOpened()

            if not success or not isOpened:
                # skip if no frame
                self.CamMan_singleton.put_user_frame(empty_frame)
                continue

            frame = cv2.rotate(cv2.flip(frame, 1), cv2.ROTATE_90_CLOCKWISE)

            if self.CamMan_singleton.calib:  # check if calibration is toggled in the user's cam thread
                frame, _ = do_calib(edge_detector, frame, self.mouse_location)
                # if in calibration, update frame and edge information

            self.CamMan_singleton.put_user_frame(frame)
            # logging.debug(str(frameClass.edge_line))

            if self.CamMan_singleton.check_Term():
                break

        print(Params.OKGREEN + "Exiting UserCam" + previewName + Params.ENDC)
        self.user_cam.release()

    def videoPreview(self, previewName, camID):
        # for video Demo, probably never used
        cam = cv2.VideoCapture(vids[camID])
        ed = edge_detection.EdgeDetection()
        frameClass = Frame(camID)
        autoResize = AutoResize()
        frame_counter = 0
        calib_length = 50

        # # demo video calib phase
        # while True:
        #     success, frame = cam.read()
        #
        #     if not success:  # check if video is played to the end
        #         cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #         # restart the video from the first frame if it is ended
        #         continue
        #
        #     image = cv2.resize(frame, (Params.VID_W, Params.VID_H))
        #     if frame_counter < calib_length:
        #         ratio = autoResize.resize(image, 100)
        #
        #         adjust_w = round(Params.VID_W * ratio)
        #         adjust_h = round(Params.VID_H * ratio)
        #         new_shape = (adjust_w, adjust_h)
        #         if ratio > 1:
        #             rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
        #             ah, aw = rsz_image.shape[:2]
        #             dn = round(ah * 0.5 + Params.VID_H * 0.5)
        #             up = round(ah * 0.5 - Params.VID_H * 0.5)
        #             lt = round(aw * 0.5 - Params.VID_W * 0.5)
        #             rt = round(aw * 0.5 + Params.VID_W * 0.5)
        #             rsz_image = rsz_image[up:dn, lt:rt]
        #         else:
        #             rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
        #
        #         # print("cam" + str(camID) + " ratio: " + str(ratio))
        #         frame_counter += 1
        #         edge = ed.process_frame(rsz_image, sample_size=100)
        #         a, b = edge
        #         if a is not None and b is not None:
        #             h, w, c = rsz_image.shape
        #             cv2.line(rsz_image, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)
        #             # cv2.imshow("test"+str(camID), resized_frame)
        #         else:
        #             edge = (0, 0)
        #         frameClass.updateFrame(image=rsz_image, edge_line=edge, ref_ratio=ratio)  # update edge information
        #     else:
        #         break
        #
        #     CamMan.put_frame(camID=camID, FRAME=frameClass)
        #     cv2.waitKey(15)
        #     if CamMan.check_Term():
        #         break
        #
        # # extract the calibrated parameters for cropping image
        # ratio = frameClass.ref_ratio
        # adjust_w = round(Params.VID_W * ratio)
        # adjust_h = round(Params.VID_H * ratio)
        # new_shape = (adjust_w, adjust_h)
        #
        # # demo video running phase
        # while True:
        #     success, frame = cam.read()
        #
        #     if not success:  # check if video is played to the end
        #         cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #         # restart the video from the first frame if it is ended
        #         continue
        #
        #     image = cv2.resize(frame, (Params.VID_W, Params.VID_H))
        #
        #     if ratio > 1:
        #         rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
        #         ah, aw = rsz_image.shape[:2]
        #         dn = round(ah * 0.5 + Params.VID_H * 0.5)
        #         up = round(ah * 0.5 - Params.VID_H * 0.5)
        #         lt = round(aw * 0.5 - Params.VID_W * 0.5)
        #         rt = round(aw * 0.5 + Params.VID_W * 0.5)
        #         rsz_image = rsz_image[up:dn, lt:rt]
        #     else:
        #         rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
        #
        #     frameClass.updateFrame(image=rsz_image)  # update image only
        #     CamMan.put_frame(camID=camID, FRAME=frameClass)
        #
        #     cv2.waitKey(15)
        #     if CamMan.check_Term():
        #         break
        #
        # print("Exiting " + previewName)
        # cam.release()


def stackParam(cam_dict, bg_shape: int):
    # should be called everytime a new cam joined
    # return: fit_width, number of image to stack, spacing for each camera,
    bg_h, bg_w, bg_c = bg_shape
    num_of_cam = len(cam_dict)
    tallest = 0
    if num_of_cam:  # check if the frame dict is empty
        for frameClass in cam_dict.values():
            h, w, c = frameClass.img.shape
            ratio = h / w
            if ratio > tallest:
                tallest = ratio

        fit_width = w_step = int(np.floor(bg_w / num_of_cam))

        fit_height = np.floor(fit_width * tallest)

        if fit_height > bg_h:
            fit_width = np.floor(bg_h / tallest)
            fit_height = bg_h

        fit_shape = (int(fit_height), int(fit_width))

        margin_h = np.floor((bg_h - fit_height)*0.5)
        margin_w = np.floor((w_step - fit_width)*0.5)
        margins = [int(margin_h), int(margin_w)]
    else:
        # if no cam need to be processed, set all parameter to zero
        # since these parameters are not need for stackIMG
        fit_shape = w_step = margins = 0
    return fit_shape, w_step, margins


class VideoJoint:
    def __init__(self):
        self.CamMan_singleton = CamManagement.CamManagement()
        self.ht = HeadTrack.HeadTrack()
        self.Q_FrameForDisplay = Queue(maxsize=3)
        self.Q_userFrame = Queue(maxsize=3)
        self.mouse_location_FE = None

        self.calib = False
        self.Term = False
        self.update_user_cam_id = threading.Event()
        self.user_cam_id = 0

        self.server = video_server.VideoServer(Params.HOST_IP, Params.PORT)

    def run(self):
        server_thread = threading.Thread(target=self.server.start, args=())
        server_thread.setDaemon(True)
        server_thread.start()
        logging.info("Socket Acceptor Started!")

        thread0 = threading.Thread(target=self.ctlThread, args=())
        thread0.setDaemon(True)
        thread0.start()
        logging.info("Control Thread Started!")
        # thread_non_block = threading.Thread(target=self.client_acceptor, args=(Params.HOST_IP, Params.PORT))
        # thread_non_block.start()
        # logging.info("Socket Acceptor Started!")

    def update_user_cam_FE(self, camID):
        while self.update_user_cam_id.is_set():
            time.sleep(0.1)
        self.update_user_cam_id.set()
        self.user_cam_id = camID

    def ctlThread(self):
        ht = self.ht

        userCam = SERVER_CAM_ID
        fit_shape, w_step, margins = 0, 0, 0
        cam_loaded = 0

        # calib_window = 'calibration window'

        imgBG_path = root_path + '/assets/background/background_demo_1.jpg'
        imgBG = cv2.imread(imgBG_path)
        imgBG = cv2.resize(imgBG, Params.BG_DIM)

        # Opens User's Camera
        cam_name = "Camera %s" % str(userCam)
        camThread = CamThread(cam_name, camID=userCam, if_usercam=True, if_demo=False)
        camThread.start()

        a_rsz = AutoResize()

        while not self.Term:
            frame_dict = self.CamMan_singleton.get_frames()

            cam_count = len(frame_dict.keys())  # get the number of camera connected
            if cam_count != cam_loaded:  # update the stack parameter everytime a new cam joined or left
                cam_loaded = cam_count
                fit_shape, w_step, margins = stackParam(frame_dict, imgBG.shape)

            if self.update_user_cam_id.is_set():
                userCam = self.user_cam_id
                self.update_user_cam_id.clear()
                camThread.set_user_cam(userCam)

            user_frame = self.CamMan_singleton.get_user_frame()
            if self.CamMan_singleton.calib:
                camThread.mouse_location = self.mouse_location_FE
                self.Q_userFrame.put(user_frame)

            imgStacked = bgmp.stackIMG(frame_dict, imgBG, fit_shape, w_step, margins)

            imgBG_output = ht.HeadTacker(cv2.flip(user_frame, 1), imgStacked, hist=10)
            a_rsz.check_bound(user_frame, imgBG_output)
            self.Q_FrameForDisplay.put(imgBG_output)
            # print("put")

        print(Params.OKGREEN + "Closing Cam Control Thread" + Params.ENDC)
        self.CamMan_singleton.set_Term(True)  # inform each camera threads to terminate
        self.CamMan_singleton.dump_frame_queue()  # dump the frame queue

    def stop(self):
        self.server.stop()
        self.Term = True

        print("!!Dumping main window queue!!")
        while not self.Q_FrameForDisplay.empty():
            item = self.Q_FrameForDisplay.get()
            print("dequeued one item")

        print("The main window queue is empty.")

        print("!!Dumping selfie queue!!")

        while not self.Q_userFrame.empty():
            item = self.Q_userFrame.get()
            print("dequeued one item")

        print("The selfie queue is empty.")

        for thread in threading.enumerate():
            if thread is not threading.currentThread():
                if thread.is_alive():
                    print("Joining Thread: ", thread)
                    thread.join(timeout=3)
                    print(thread, " Joined")
