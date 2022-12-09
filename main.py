# Import cvzone (opencv-python must be in 4.5.5.62), mediapipe
import cv2
import numpy as np
import threading
import time
import logging
import HeadTrack
import bg_remove_mp
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

vid1 = "vid/demo2.mp4"
vid2 = "vid/demo1.MOV"
vid3 = cv2.VideoCapture("vid/IMG_0582.MOV")
vids = {101: vid1, 102: vid2}

# mp_drawing = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh


# def Yshift_img(img_vector, y_offset: int, fill_clr=(0, 0, 0)):
#     # y_offset: positive: upward shifting; negative: downward shifting
#
#     h, w, c = img_vector.shape
#
#     blank = np.full(shape=(np.abs(y_offset), w, c), fill_value=fill_clr)
#     if y_offset > 0:
#         stack_img = np.vstack((img_vector, blank))
#         h1, w1, c1 = stack_img.shape
#         h = h1 - h
#         img_out = stack_img[h:h1, 0:w, :]
#     else:
#         stack_img = np.vstack((blank, img_vector))
#         img_out = stack_img[0:h, 0:w, :]
#
#     return img_out


def stackParam(cam_dict, bg_shape: int):
    # should be called everytime a new cam joined
    # return: fit_width, number of image to stack, spacing for each camera,
    bg_h, bg_w, bg_c = bg_shape
    num_of_cam = len(cam_dict)
    tallest = 0
    for frame in cam_dict.values():
        h, w, c = frame.shape
        ratio = h/w
        if ratio > tallest:
            tallest = ratio

    fit_width = w_step = int(np.floor(bg_w / num_of_cam))

    fit_height = np.floor(fit_width * tallest)

    if fit_height > bg_h:
        fit_width = np.floor(bg_h / tallest)
        fit_height = bg_h

    fit_shape = (int(fit_height), int(fit_width))

    margin_h = np.floor((bg_h - fit_height) / 2)
    margin_w = np.floor((w_step - fit_width) / 2)
    margins = [int(margin_h), int(margin_w)]

    return fit_shape, w_step, margins


def stackIMG(cam_dict, bg_img, fit_shape, w_step, margins, cam_shift_y):
    loc_bgIMG = bg_img.copy()
    fit_h, fit_w = fit_shape
    bg_h, bg_w, bg_c = bg_img.shape
    h_margin, w_margin = margins[0], margins[1]
    i = 0

    for camID in cam_dict:
        frame = cam_dict[camID]
        rsz_cam = cv2.resize(frame, (fit_w, fit_h))
        # rsz_cam = Yshift_img(rsz_cam, cam_shift_y[camID], BLUE)
        shift_y = cam_shift_y[camID]  # extract the shift value on y-axis
        print(str(camID)+": shift_y = "+str(shift_y))
        top_spacing = h_margin + shift_y
        x_left = w_step * i + w_margin
        x_right = x_left + fit_w
        # y_top = clamp(h_margin, 0, bg_h)
        # y_bottom = clamp(h_margin + fit_h, 0, bg_h)
        y_top = h_margin
        half = round(h_margin/2)
        y_bottom = h_margin + fit_h
        if top_spacing <= 0:
            a = rsz_cam[0:fit_h+top_spacing, :, :]
            loc_bgIMG[-top_spacing:fit_h, x_left:x_right, :] = a
        else:
            b = rsz_cam[top_spacing:fit_h, :, :]
            loc_bgIMG[0:fit_h-top_spacing, x_left:x_right, :] = b
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

    def open_cam(self, camID=cam_id, if_user=False, if_demo=False):
        cam_name = "Camera %s" % str(camID)
        camThread = CamThread(cam_name, camID, if_user, if_demo)
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
            if camID in self.edge_y.keys():  # check if the cam id associated with an edge information
                # convert the edge location to the offset in background's dimension
                background_y = np.floor(self.edge_y[camID]*BG_H/CAM_H)
                # convert offset to background coordinate
                offset[camID] = int(background_y - self.reference_y)
                # postive: table belows ref line, negative: table above ref line
            else:
                offset[camID] = 0
        return offset

    def toggle_calib(self, camID):
        self.calib = not self.calib


class CamThread(threading.Thread):
    def __init__(self, previewName, camID, if_usercam, if_demo):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.if_usercam = if_usercam
        self.if_demo = if_demo

    def run(self):
        print("Starting Thread" + self.previewName)
        # segmentor = SelfiSegmentation()  # setup BGremover
        if self.if_demo:
            videoPreview(self.previewName, self.camID)
        else:
            camPreview(self.previewName, self.camID, self.if_usercam)


def videoPreview(previewName, camID):
    cam = cv2.VideoCapture(vids[camID])
    ed = edge_detection.EdgeDetection()
    frame_counter = 0
    calib_length = 50
    seg_bg = bg_remove_mp.SegmentationBG()
    while True:
        success, frame = cam.read()
        if not success:
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # restart the video from the first frame if it is ended
            continue
        else:
            resized_frame = cv2.resize(frame, (360, 640))
            if frame_counter < calib_length:
                frame_counter += 1
                edge = ed.process_frame(resized_frame, threshold=100)
                a, b = edge
                CamMan.save_edge(camID, [a, b])
                # CamMan.save_edge(camID, [a, b])
                if a is not None and b is not None:
                    h, w, c = resized_frame.shape
                    cv2.line(resized_frame, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)
                    # cv2.imshow("test"+str(camID), resized_frame)

            blurred_output = seg_bg.blur_bg(resized_frame)
            CamMan.save_frame(camID=camID, frame=blurred_output)

        cv2.waitKey(15)
        if CamMan.check_Term():
            break

    print("Exiting " + previewName)
    cam.release()


def camPreview(previewName, camID, if_usercam):
    # cam = vids[camID]
    # cv2.namedWindow("iso frame " + str(camID))
    # Real time video cap
    cam = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    ed = edge_detection.EdgeDetection()
    cam.set(3, 640)  # width
    cam.set(4, 360)  # height


    while True:
        success, frame = cam.read()

        if not success:
            # skip if no frame
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if CamMan.calib and if_usercam:  # check if calibration is toggled in the user's cam thread
            edge = ed.process_frame(frame, threshold=100)
            a, b = edge
            CamMan.save_edge(camID, [a, b])
            if a is not None and b is not None:
                h, w, c = frame.shape
                cv2.line(frame, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)


        CamMan.save_frame(camID=camID, frame=frame)
        logging.debug(str(CamMan.get_edge()))

        if CamMan.check_Term():
            break

    print("Exiting " + previewName)
    cam.release()


def ctlThread():
    userCam = 0
    clientCam = 0
    fit_shape, w_step, margins = 0, 0, 0
    cam_loaded = 0

    name = "Video"
    calib_window = 'calibration window'
    cv2.namedWindow(name)

    imgBG = cv2.imread("background/graphicstock-blurred-bokeh-interior-background-image-in-whites_B-3xL3NnluW_thumb.jpg")
    imgBG = cv2.resize(imgBG, BG_DIM)

    CamMan.open_cam(camID=userCam, if_user=True)
    # CamMan.open_cam(camID=clientCam)
    CamMan.open_cam(camID=101, if_demo=True)
    CamMan.open_cam(camID=102, if_demo=True)

    while True:
        frame_dict = CamMan.get_frame()

        if not frame_dict:
            continue  # if frame dictionary is empty, continue
        user_feed = frame_dict[userCam]
        if CamMan.calib:  # if calibration is toggled by user
            cv2.imshow(calib_window, user_feed)

        frame_dict.pop(userCam, None)  # user camera won't be displayed

        cam_count = len(frame_dict.keys())  # get the number of camera connected
        if cam_count != cam_loaded:  # update the stack parameter everytime a new cam joined
            cam_loaded = cam_count
            fit_shape, w_step, margins = stackParam(frame_dict, imgBG.shape)

        cam_offset_y = CamMan.get_cam_offset()
        if cam_offset_y:
            imgStacked = stackIMG(frame_dict, imgBG, fit_shape, w_step, margins, cam_offset_y)
        else:
            for cam in frame_dict:
                cam_offset_y[cam] = 0
            imgStacked = stackIMG(frame_dict, imgBG, fit_shape, w_step, margins, cam_offset_y)

        # temp = np.subtract(imgStacked, BLUE)
        #
        # # Transparent mask stores boolean value
        # mask = (temp == (0, 0, 0))
        # mask_singleCH = (mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2])
        #
        # alpha = np.zeros(imgStacked.shape, dtype=np.uint8)
        # imgBG_output = imgBG.copy()temp = np.subtract(imgStacked, BLUE)
        #
        # # Transparent mask stores boolean value
        # mask = (temp == (0, 0, 0))
        # mask_singleCH = (mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2])
        #
        # alpha = np.zeros(imgStacked.shape, dtype=np.uint8)
        # imgBG_output = imgBG.copy()
        # # imgBG_output[mask_bin, :] = imgBG[mask_bin, :]
        # imgBG_output[~mask_singleCH, :] = imgStacked[~mask_singleCH, :]
        #
        # alpha[mask_singleCH] = 0
        # alpha[~mask_singleCH] = 255
        # alpha_grey = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        #
        # contours, hierarchy = cv2.findContours(alpha_grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # image_copy = imgBG_output.copy()
        # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
        #                  lineType=cv2.LINE_AA)
        # blurred_img = cv2.GaussianBlur(imgBG_output, (9, 9), 0)
        # output = np.where(mask == np.array([0, 255, 0]), blurred_img, imgBG_output)
        # output = np.where(imgStacked == np.array([255, 0, 0]), imgBG, imgStacked)
        ref_y = round(BG_H*6/7)
        cv2.line(imgStacked, (0, ref_y), (BG_W, ref_y), (255, 255, 0))
        imgBG_output = ht.HeadTacker(user_feed, imgStacked, hist=10)
        cv2.imshow(name, imgBG_output)

        # cv2.imshow("Test", alpha_grey)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('t'):
            CamMan.toggle_calib(userCam)
            if not CamMan.calib:  # check if calib is toggled to false
                ht.set_calib()
                cv2.destroyWindow(calib_window)

    CamMan.set_Term(True)
    cv2.destroyWindow(name)


CamMan = CamManagement()
ht = HeadTrack.HeadTrack()
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
