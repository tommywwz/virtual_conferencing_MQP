import numpy as np
import threading
from Utils import Params
from queue import Queue
from Utils.Frame import Frame


# logging.basicConfig(level=logging.DEBUG)
FRAMES_lock = threading.Lock()
camDEL_lock = threading.Lock()


class CamManagement:
    # cam_id = 0
    reference_y = np.floor(Params.BG_DIM[1] * 6 / 7)

    def __init__(self):
        self.FRAMES_dictQ = {}  # a dictionary that holds a queue of Frame data structure
        self.TERM = False
        # self.edge_lines = {}  # edge equation (a, b) for each cam
        # self.edge_y = {}  # average height of edge for each cam
        self.empty_frame = np.zeros(Params.SHAPE, dtype=np.uint8)
        self.calib = True
        self.calibCam = None

    # def open_cam(self, camID=cam_id, if_user=False, if_demo=False):
    #     cam_name = "Camera %s" % str(camID)
    #     camThread = CamThread(cam_name, camID, if_user, if_demo)
    #     camThread.start()
    #     time.sleep(0.5)
    #     logging.info("%s: starting", cam_name)
    #     # self.FRAMES[camID] = self.empty_frame
    #     # self.edge_lines[camID] = [None, None]
    #     self.cam_id += 1  # todo camera conflicts need to be fixed here
    #     return True

    def init_cam(self, camID, queue_size=3):
        # initialize a queue for the given camID
        # !your must init a frame queue in the dictionary to put and get frames!
        with FRAMES_lock:
            self.FRAMES_dictQ[camID] = Queue(maxsize=queue_size)

    def put_frame(self, camID, FRAME):
        # put a FRAME class in the queue
        # !our must init a frame queue in the dictionary to put and get frames!
        # No need to lock here
        self.FRAMES_dictQ[camID].put(FRAME)

    def get_frames(self):
        # !your must init a frame queue in the dictionary to put and get frames!
        frame_dict = {}
        with FRAMES_lock:
            with camDEL_lock:
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
                    item = frame_queue.get()
                    print("dequeued one item")
            else:
                print("The queue is empty.")

    def delete_cam(self, camID):
        flag_frame = Frame(camID)
        flag_frame.close = True  # genrate a frame to
        self.FRAMES_dictQ[camID].put(flag_frame)
        with camDEL_lock:
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

