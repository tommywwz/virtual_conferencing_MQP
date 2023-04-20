# reference: https://pyshine.com/Socket-programming-and-openc/
import cv2
import pickle
import socket
import struct
import threading
import queue
import numpy as np
from Utils import edge_detection, Params
from Utils.Frame import Frame
from Utils.AutoResize import AutoResize

buff_4K = 4 * 1024


def gen_fake_frame():
    blank_frame = np.zeros((Params.VID_H, Params.VID_W, 3), dtype=np.uint8)
    fake_edge = (0, 0)
    return blank_frame, fake_edge


def put_text_on_center(frame, text, color=(0, 0, 255), font_scale=1, thickness=2):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_w, text_h = text_size
    frame_h, frame_w = frame.shape[:2]
    cv2.putText(frame, text, (int((frame_w - text_w) / 2), int((frame_h - text_h) / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)
    # cv2.putText(frame,
    #             text='Calibrating',
    #             org=int((frame_w - text_w) / 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=color,
    #             thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
    return frame


cam_mutex = threading.Lock()


class ClientVideo(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

        self.PORT = 9999
        self.HOST_IP = '192.168.1.3'  # paste your server ip address here

        self.cam = None
        self.old_cam_id = 0
        self.set_cam(self.old_cam_id)

        self.close_lock = threading.Lock()
        self.client_socket = None
        self.Q_selfie = queue.Queue(maxsize=3)
        self.edge_line = (0, 0)
        self.resize_ratio = 1
        self.new_shape = (Params.VID_W, Params.VID_H)

        self.calib_flag = False
        self.exit_flag = False
        self.server_down = threading.Event()

        self.mouse_location = None  # location of user mouse click

        self.edge_detector = edge_detection.EdgeDetection()
        self.client_auto_resize = AutoResize()

        client_id = socket.gethostbyname(socket.gethostname())
        print("HOST NAME: ", client_id)
        self.frameClass = Frame(client_id)

    def set_cam(self, camID):

        if self.cam is not None and camID == self.old_cam_id:
            return
        self.cam = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
        self.cam.set(3, Params.RAW_CAM_W)  # width
        self.cam.set(4, Params.RAW_CAM_H)  # height
        self.old_cam_id = camID
        # check if the camera is opened
        if self.cam.isOpened():
            pass
        else:
            raise IOError("Cannot open webcam")

    def set_connection(self, IP, port=9999):
        self.HOST_IP = IP
        self.PORT = port
        # create socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.settimeout(5)
        try:
            self.client_socket.connect((self.HOST_IP, self.PORT))  # a tuple
        except socket.timeout:
            raise socket.timeout
        except socket.error as e:
            raise e

    def run(self):
        if self.exit_flag:
            exit(0)  # exit the thread

        while True:
            with self.close_lock:
                if self.exit_flag: break

                loc_cam = self.cam

                if self.calib_flag:
                    if loc_cam.isOpened():
                        success, frame = loc_cam.read()
                        if success:
                            frame = cv2.rotate(cv2.flip(frame.copy(), 1), cv2.ROTATE_90_CLOCKWISE)  # rotate raw frame
                            self.do_calibration(frame)
                    else:
                        frame, _ = gen_fake_frame()
                        frame = put_text_on_center(frame, "Camera not found", color=(0, 0, 255), font_scale=1,
                                                   thickness=2)
                        self.Q_selfie.put(frame)

                    # when calibrating, sends blank image to server
                    blank_screen_for_server, fake_edge = gen_fake_frame()
                    blank_screen_for_server = put_text_on_center(blank_screen_for_server, "Calibrating...",
                                                                 color=(0, 0, 255), font_scale=3)
                    self.frameClass.updateFrame(image=blank_screen_for_server, edge_line=fake_edge)
                    try:
                        self.send_msg(self.frameClass)
                    except ConnectionResetError or ConnectionError as e:
                        print(e)
                        self.calib_flag = False
                        continue
                    continue

                if loc_cam.isOpened():
                    success, frame = loc_cam.read()
                    if success:
                        frame = cv2.rotate(cv2.flip(frame.copy(), 1), cv2.ROTATE_90_CLOCKWISE)  # rotate raw frame
                else:
                    frame, _ = gen_fake_frame()
                    frame = put_text_on_center(frame, "Camera not found", color=(0, 0, 255), font_scale=1, thickness=2)
                    self.Q_selfie.put(frame)
                    continue

                if self.client_socket is None:
                    self.Q_selfie.put(frame)
                    continue

                rsz_image = self.do_resize(frame)
                self.Q_selfie.put(frame)

                self.frameClass.updateFrame(image=rsz_image, edge_line=self.edge_line)  # update edge information
                try:
                    self.send_msg(self.frameClass)
                except ConnectionResetError or ConnectionError as e:
                    print(e)
                    continue

    def send_msg(self, frameClass):
        if self.client_socket is None:
            raise ConnectionError("Client socket is None")
        try:
            pickled_frame = pickle.dumps(frameClass)
            # data length followed by serialized frame object
            msg = struct.pack("Q", len(pickled_frame)) + pickled_frame
            self.client_socket.sendall(msg)
        except ConnectionResetError as e:
            self.server_down.set()
            self.client_socket = None
            raise e

    def do_resize(self, frame):
        # -------------using the resizing ratio to resize image----------------
        if self.resize_ratio > 1:
            # if head is smaller than reference, the photo need to be enlarged
            rsz_image = cv2.resize(frame, self.new_shape, interpolation=cv2.INTER_LINEAR)
            ah, aw = rsz_image.shape[:2]
            dn = round((ah + Params.VID_H) * 0.5)
            up = round((ah - Params.VID_H) * 0.5)
            lt = round((aw - Params.VID_W) * 0.5)
            rt = round((aw + Params.VID_W) * 0.5)
            rsz_image = rsz_image[up:dn, lt:rt]
            # gets an image the same as the standard shape
        else:
            # if head is bigger than reference, the photo need to be shrunk
            rsz_image = cv2.resize(frame, self.new_shape, interpolation=cv2.INTER_AREA)
            # gets a smaller image here
        # -------------end of image resizing----------------
        return rsz_image

    def do_calibration(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        linetype = cv2.LINE_AA
        # calculate the resizing ratio
        self.resize_ratio = self.client_auto_resize.resize(frame, 100)
        adjust_w = round(Params.VID_W * self.resize_ratio)
        adjust_h = round(Params.VID_H * self.resize_ratio)
        self.new_shape = (adjust_w, adjust_h)

        # calculate real edge
        self.edge_line = self.edge_detector.process_frame(frame, sample_size=100, prefer_point=self.mouse_location)
        cv2.circle(frame, self.mouse_location, radius=3, color=(255, 0, 255), thickness=-1)
        cv2.putText(frame,
                    text='Calibrating',
                    org=(Params.VID_W - 120, 20), fontFace=font, fontScale=.7, color=(0, 0, 255),
                    thickness=1, lineType=linetype, bottomLeftOrigin=False)

        cv2.putText(frame,
                    text='Please make sure your hands are below the table',
                    org=(10, Params.VID_H - 10), fontFace=font, fontScale=.4, color=(0, 0, 255),
                    thickness=1, lineType=linetype, bottomLeftOrigin=False)

        if self.edge_line is None:  # invalid edge line
            cv2.putText(frame,
                        text='Edge is not detected!',
                        org=(10, Params.VID_H - 30), fontFace=font, fontScale=.4, color=(0, 0, 255),
                        thickness=1, lineType=linetype, bottomLeftOrigin=False)
            self.Q_selfie.put(frame)
            return

        a, b = self.edge_line

        if self.resize_ratio < 1:
            # if the image is shrunk, adjust the edge location accordingly
            self.edge_line = round(a * self.resize_ratio), round(b * self.resize_ratio)

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

        self.Q_selfie.put(frame)
        return

    def toggle_calib(self):
        self.calib_flag = not self.calib_flag

    def get_Queue(self):
        return self.Q_selfie.get()

    def dump_Queue(self):
        while not self.Q_selfie.empty():
            item = self.Q_selfie.get()
            print("dequeued one item")

        print("The queue is empty.")

    def close(self):
        self.exit_flag = True
        self.dump_Queue()
        with self.close_lock:
            if self.cam.isOpened():
                self.cam.release()
            if self.client_socket is not None:
                self.client_socket.close()
        print("Backend Thread Exited")


if __name__ == "__main__":
    thread0 = ClientVideo()
    # thread1 = threading.Thread(target=audio_stream)
    thread0.start()
    # thread1.start()
    print("starting threads")
    thread0.join()
    # thread1.join()

    print("All threads are terminated")
    exit(0)
