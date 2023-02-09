# Import cvzone (opencv-python must be in 4.5.5.62), mediapipe
import os
import cv2
import numpy as np
import threading
import time
import logging
import struct, select, pickle, Params, socket
from queue import Queue
from wheels import bg_remove_mp as bgmp, edge_detection, HeadTrack
from wheels.AutoResize import AutoResize
from wheels.Frame import Frame

logging.basicConfig(level=logging.DEBUG)


current_path = os.path.dirname(__file__)
root_path = os.path.split(current_path)[0]
vid1 = "assets/vid/demo2.mp4"
vid2 = "assets/vid/demo1.mp4"
vid3 = "assets/vid/demo3.mp4"
vid4 = "assets/vid/demo4.mp4"
vid5 = "assets/vid/demo5.mp4"
vid6 = "assets/vid/demo6.mp4"
vids = {101: vid1, 102: vid2, 103: vid3, 104: vid4, 105: vid5, 106: vid6}


def stackParam(cam_dict, bg_shape: int):
    # should be called everytime a new cam joined
    # return: fit_width, number of image to stack, spacing for each camera,
    bg_h, bg_w, bg_c = bg_shape
    num_of_cam = len(cam_dict)
    tallest = 0
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

    return fit_shape, w_step, margins


class CamManagement:
    cam_id = 0
    reference_y = np.floor(Params.BG_DIM[1] * 6 / 7)

    def __init__(self):
        self.FRAMES = {}  # a dictionary that holds a queue of Frame data structure
        self.TERM = False
        self.edge_lines = {}  # edge equation (a, b) for each cam
        self.edge_y = {}  # average height of edge for each cam
        self.empty_frame = np.zeros(Params.SHAPE, dtype=np.uint8)
        self.calib = True
        self.calibCam = None

    def open_cam(self, camID=cam_id, if_user=False, if_demo=False):
        cam_name = "Camera %s" % str(camID)
        camThread = CamThread(cam_name, camID, if_user, if_demo)
        camThread.start()
        time.sleep(0.5)
        logging.info("%s: starting", cam_name)
        # self.FRAMES[camID] = self.empty_frame
        # self.edge_lines[camID] = [None, None]
        self.cam_id += 1  # todo camera conflicts need to be fixed here
        return True

    def init_frame(self, camID, queue_size=3):
        # initialize a queue for the given camID
        # !your must init a frame queue in the dictionary to put and get frames!
        self.FRAMES[camID] = Queue(maxsize=queue_size)

    def put_frame(self, camID, frame):
        # put a frame in the queue
        # !our must init a frame queue in the dictionary to put and get frames!
        self.FRAMES[camID].put(frame)

    def get_frames(self):
        # !your must init a frame queue in the dictionary to put and get frames!
        frame_dict = {}
        for camID in self.FRAMES:
            frame_dict[camID] = self.FRAMES[camID].get()
            # extract frame queue by key and save an item from the queue to output dictionary
        return frame_dict

    def delete_cam(self, camID):
        self.FRAMES.pop(camID, None)

    def set_Term(self, ifTerm: bool):
        self.TERM = ifTerm

    def check_Term(self):
        return self.TERM

    # def save_edge(self, camID, edge):
    #     a, b = edge
    #     if a is not None and b is not None:
    #         self.edge_lines[camID] = edge
    #         self.edge_y[camID] = int(np.floor(RAW_CAM_W * a / 2 + b))  # height of edge's midpoint
    #
    # def get_edge(self):
    #     return self.edge_lines
    #
    # def get_edge_y(self):
    #     return self.edge_y

    def toggle_calib(self):
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
        if self.if_demo:  # if starting a demo video, starts videoPreview, otherwise, starts camera thread
            videoPreview(self.previewName, self.camID)
        else:
            camPreview(self.previewName, self.camID, self.if_usercam)


class VideoClientThread(threading.Thread):
    def __init__(self, client_socket, client_addr):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.client_addr = client_addr

    def run(self):
        clientThread(self.client_socket, self.client_addr)


def clientThread(client_socket, client_addr):
    inputs = [client_socket]
    cltAddr_camID, clt_port = client_addr
    data = b""
    payload_size = struct.calcsize("Q")
    CamMan.init_frame(cltAddr_camID)  # initialize the FIFO queue for current camera feed
    while True:
        readable, writable, exceptional = select.select(inputs, [], inputs)
        if exceptional:
            # The client socket has been closed abruptly
            client_socket.close()
            inputs.remove(client_socket)
            print(str(client_addr) + ": abruptly exit")
            break

        while len(data) < payload_size:
            packet = client_socket.recv(Params.buff_4K)  # 4K
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]  # extracting the packet size information

        if not packed_msg_size:  # check if client has lost connection
            client_socket.close()
            inputs.remove(client_socket)
            print("Client:", client_addr, " Exited")
            break

        data = data[payload_size:]  # extract the img data from the rest of the packet
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            # keep loading the data until the entire data received
            data += client_socket.recv(Params.buff_4K)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frameClass = pickle.loads(frame_data)
        CamMan.put_frame(cltAddr_camID, frameClass)

    CamMan.delete_cam(cltAddr_camID)


def videoPreview(previewName, camID):
    # for video Demo, probably never used
    cam = cv2.VideoCapture(vids[camID])
    ed = edge_detection.EdgeDetection()
    frameClass = Frame(camID)
    autoResize = AutoResize()
    frame_counter = 0
    calib_length = 50

    while True:  # demo video calib phase
        success, frame = cam.read()

        if not success:  # check if video is played to the end
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # restart the video from the first frame if it is ended
            continue

        image = cv2.resize(frame, (Params.VID_W, Params.VID_H))
        if frame_counter < calib_length:
            ratio = autoResize.resize(image, 100)

            adjust_w = round(Params.VID_W * ratio)
            adjust_h = round(Params.VID_H * ratio)
            new_shape = (adjust_w, adjust_h)
            if ratio > 1:
                rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
                ah, aw = rsz_image.shape[:2]
                dn = round(ah * 0.5 + Params.VID_H * 0.5)
                up = round(ah * 0.5 - Params.VID_H * 0.5)
                lt = round(aw * 0.5 - Params.VID_W * 0.5)
                rt = round(aw * 0.5 + Params.VID_W * 0.5)
                rsz_image = rsz_image[up:dn, lt:rt]
            else:
                rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)

            # print("cam" + str(camID) + " ratio: " + str(ratio))
            frame_counter += 1
            edge = ed.process_frame(rsz_image, threshold=100)
            a, b = edge
            if a is not None and b is not None:
                h, w, c = rsz_image.shape
                cv2.line(rsz_image, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)
                # cv2.imshow("test"+str(camID), resized_frame)
            else:
                edge = (0, 0)
            frameClass.updateFrame(image=rsz_image, edge_line=edge, ref_ratio=ratio)  # update edge information
        else:
            break

        CamMan.put_frame(camID=camID, frame=frameClass)
        cv2.waitKey(15)
        if CamMan.check_Term():
            break

    # extract the calibrated parameters for cropping image
    ratio = frameClass.ref_ratio
    adjust_w = round(Params.VID_W * ratio)
    adjust_h = round(Params.VID_H * ratio)
    new_shape = (adjust_w, adjust_h)

    while True:  # demo video running phase
        success, frame = cam.read()

        if not success:  # check if video is played to the end
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # restart the video from the first frame if it is ended
            continue

        image = cv2.resize(frame, (Params.VID_W, Params.VID_H))

        if ratio > 1:
            rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
            ah, aw = rsz_image.shape[:2]
            dn = round(ah * 0.5 + Params.VID_H * 0.5)
            up = round(ah * 0.5 - Params.VID_H * 0.5)
            lt = round(aw * 0.5 - Params.VID_W * 0.5)
            rt = round(aw * 0.5 + Params.VID_W * 0.5)
            rsz_image = rsz_image[up:dn, lt:rt]
        else:
            rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)

        frameClass.updateFrame(image=rsz_image)  # update image only
        CamMan.put_frame(camID=camID, frame=frameClass)

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
    cam.set(3, Params.RAW_CAM_W)  # width
    cam.set(4, Params.RAW_CAM_H)  # height
    frameClass = Frame(camID)
    CamMan.init_frame(camID)

    while True:
        success, frame = cam.read()

        if not success:
            # skip if no frame
            continue

        frame = cv2.rotate(cv2.flip(frame, 1), cv2.ROTATE_90_CLOCKWISE)

        if CamMan.calib and if_usercam:  # check if calibration is toggled in the user's cam thread
            edge = ed.process_frame(frame, threshold=100)
            a, b = edge
            font = cv2.FONT_HERSHEY_SIMPLEX
            linetype = cv2.LINE_AA
            cv2.putText(frame,
                        text='Press T when satisfied with table edge',
                        org=(10, 30), fontFace=font, fontScale=.55, color=(0, 255, 0),
                        thickness=1, lineType=linetype, bottomLeftOrigin=False)
            cv2.putText(frame,
                        text='Please make sure your hands are below the table',
                        org=(10, Params.VID_H - 10), fontFace=font, fontScale=.4, color=(0, 0, 255),
                        thickness=1, lineType=linetype, bottomLeftOrigin=False)
            if a is not None and b is not None:
                h, w, c = frame.shape
                left_intercept = b
                right_intercept = w*a+b
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
            else:
                edge = (0, 0)

            frameClass.updateFrame(image=frame, edge_line=edge)
            # if in calibration, update frame and edge information
        else:
            frameClass.updateFrame(image=frame)  # if not in calibration, update frame only

        CamMan.put_frame(camID=camID, frame=frameClass)
        # logging.debug(str(frameClass.edge_line))

        if CamMan.check_Term():
            break

    print("Exiting " + previewName)
    cam.release()


def client_acceptor(server_addr, server_port):
    server_sock = socket.socket(socket.AF_INET,
                                socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host_ips = socket.gethostbyname_ex(host_name)
    print('host ip: ' + server_addr)
    server_socket_addr = (server_addr, server_port)
    # socket bind
    server_sock.bind(server_socket_addr)

    # socket listen
    server_sock.listen(5)
    print('Listening at: ', server_socket_addr)

    while True:
        # waiting from client connection and create a thread for it
        client_socket, client_addr = server_sock.accept()
        print('GOT NEW VIDEO CONNECTION FROM: ', client_addr)
        if client_socket:
            newClientVidThread = VideoClientThread(client_socket, client_addr)
            newClientVidThread.start()
            print("starting thread for client:", client_addr)


def ctlThread():
    userCam = 0
    fit_shape, w_step, margins = 0, 0, 0
    cam_loaded = 0

    name = "Video"
    calib_window = 'calibration window'
    cv2.namedWindow(name)

    imgBG_path = root_path + '/assets/background/background_demo_1.jpg'
    imgBG = cv2.imread(imgBG_path)
    imgBG = cv2.resize(imgBG, Params.BG_DIM)

    CamMan.open_cam(camID=userCam, if_user=True)

    a_rsz = AutoResize()

    while True:
        frame_dict = CamMan.get_frames()

        if not frame_dict:
            continue  # if frame dictionary is empty, continue
        user_feed = frame_dict[userCam].img
        frame_dict.pop(userCam, None)  # user camera won't be displayed
        if CamMan.calib:  # if calibration is toggled by user
            cv2.imshow(calib_window, user_feed)

        if not frame_dict:
            # if other clients are not connected, continue
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('t'):
                CamMan.toggle_calib()
                if not CamMan.calib:  # check if calib is toggled to false
                    ht.set_calib()  # tell head tracking method to start calibration
                    cv2.destroyWindow(calib_window)
            continue

        cam_count = len(frame_dict.keys())  # get the number of camera connected
        if cam_count != cam_loaded:  # update the stack parameter everytime a new cam joined
            cam_loaded = cam_count
            fit_shape, w_step, margins = stackParam(frame_dict, imgBG.shape)

        imgStacked = bgmp.stackIMG(frame_dict, imgBG, fit_shape, w_step, margins)

        imgBG_output = ht.HeadTacker(cv2.flip(user_feed, 1), imgStacked, hist=10)
        a_rsz.check_bound(user_feed, imgBG_output)
        cv2.imshow(name, imgBG_output)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('t'):
            CamMan.toggle_calib()
            if not CamMan.calib:  # check if calib is toggled to false
                ht.set_calib()  # tell head tracking method to start calibration
                cv2.destroyWindow(calib_window)

    CamMan.set_Term(True)
    cv2.destroyWindow(name)


CamMan = CamManagement()
ht = HeadTrack.HeadTrack()

thread0 = threading.Thread(target=ctlThread)
thread_non_block = threading.Thread(target=client_acceptor, args=(Params.HOST_IP, Params.PORT))
thread_non_block.start()
logging.info("Socket Acceptor Started!")
logging.info("Starting Control Thread")
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
