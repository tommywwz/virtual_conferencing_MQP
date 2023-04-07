import numpy as np


class Frame:
    VID_W = 360
    VID_H = 640
    VID_SHAPE = (VID_H, VID_W, 3)

    RAW_CAM_W = VID_H  # original camera setting is in landscape (will be rotated to portrait view in cam thread)
    RAW_CAM_H = VID_W  # so the camera's height & width is demo video's width & height
    RAW_CAM_SHAPE = (RAW_CAM_H, RAW_CAM_W, 3)

    def __init__(self, CamID):
        self.CamID = CamID
        self.img = np.zeros(self.VID_SHAPE, dtype=np.uint8)
        self.edge_line = None
        self.edge_y = None
        self.ref_ratio = 1
        self.close = False  # inform the thread who is getting the frame to stop receiving it

    def updateFrame(self, image, edge_line=None, ref_ratio=None):
        h, w, c = image.shape
        if edge_line is not None:
            a, b = edge_line
            self.edge_line = [a, b]
            self.edge_y = int(np.floor(self.RAW_CAM_W * a / 2 + b))  # height of edge's midpoint
        if ref_ratio is not None:
            self.ref_ratio = ref_ratio
            # self.edge_y = int(np.floor(self.edge_y * ref_ratio))
            # self.edge_line[0] = self.edge_line[0] * ref_ratio

        self.img = image

