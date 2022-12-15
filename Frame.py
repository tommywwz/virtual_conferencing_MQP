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
        self.edge_line = (0, 0)
        self.edge_y = 0
        self.ref_size = 1

    def updateFrame(self, image, edge_line=None, ref_size=None):
        self.img = image
        if edge_line is not None:
            a, b = edge_line
            if a is not None and b is not None:
                self.edge_line = edge_line
                self.edge_y = int(np.floor(self.RAW_CAM_W * a / 2 + b))  # height of edge's midpoint
        if ref_size is not None:
            self.ref_size = ref_size

