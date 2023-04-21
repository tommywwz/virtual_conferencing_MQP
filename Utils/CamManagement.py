import numpy as np
import threading
from Utils import Params
from queue import Queue
from Utils.Frame import Frame


# logging.basicConfig(level=logging.DEBUG)


class CamManagement:
    __instance = None

    reference_y = np.floor(Params.BG_DIM[1] * 6 / 7)
    FRAMES_dictQ = {}  # a dictionary that holds a queue of Frame data structure
    UserFramesQ = Queue(maxsize=3)  # user image queue
    TERM = False
    # self.edge_lines = {}  # edge equation (a, b) for each cam
    # self.edge_y = {}  # average height of edge for each cam
    empty_frame = np.zeros(Params.SHAPE, dtype=np.uint8)
    calib = False
    FRAMES_lock = threading.Lock()
    camDEL_lock = threading.Lock()
    # self.calibCam = None\

    def __new__(cls):
        if CamManagement.__instance is None:
            CamManagement.__instance = object.__new__(cls)
        return CamManagement.__instance

    def init_cam(self, camID, queue_size=3):
        # initialize a queue for the given camID
        # !your must init a frame queue in the dictionary to put and get frames!
        with self.FRAMES_lock:
            if camID in self.FRAMES_dictQ:
                print("!!Cam already exists!!")
                return
            self.FRAMES_dictQ[camID] = Queue(maxsize=queue_size)

    def put_frame(self, camID, FRAME):
        # put a FRAME class in the queue
        # !our must init a frame queue in the dictionary to put and get frames!
        # No need to lock here
        self.FRAMES_dictQ[camID].put(FRAME)

    def put_user_frame(self, frame):
        self.UserFramesQ.put(frame)

    def get_user_frame(self):
        return self.UserFramesQ.get()

    def get_frames(self):
        # !your must init a frame queue in the dictionary to put and get frames!
        frame_dict = {}
        with self.FRAMES_lock:
            with self.camDEL_lock:
                for camID in self.FRAMES_dictQ:
                    current_Frame = self.FRAMES_dictQ[camID].get()
                    if current_Frame.close:
                        continue  # if the flag frame is detected, continue and release the lock
                    else:
                        frame_dict[camID] = current_Frame
                    # extract frame queue by key and save an item from the queue to output dictionary
        return frame_dict

    def dump_frame_queue(self):
        print("!!Dumping frame queue!!")
        for frame_queue in self.FRAMES_dictQ.values():
            if not frame_queue.empty():
                while not frame_queue.empty():
                    frame_queue.get()
                    print("dequeued one item")
            else:
                print("The queue is empty.")

    def delete_cam(self, camID):
        flag_frame = Frame(camID)
        flag_frame.close = True  # genrate a frame to inform the thread to stop receiving frames of this cam
        self.FRAMES_dictQ[camID].put(flag_frame)
        with self.camDEL_lock:
            del self.FRAMES_dictQ[camID]

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

    def clear_singleton(self):
        self.__instance = None

