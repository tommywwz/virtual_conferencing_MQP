import numpy as np
import cv2


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
        self.edge_line = [0, 0]
        self.edge_y = 0
        self.ref_ratio = 1

    def updateFrame(self, image, edge_line=None, ref_ratio=None):
        h, w, c = image.shape
        if edge_line is not None:
            a, b = edge_line
            if a is not None and b is not None:
                self.edge_line = [a, b]
                self.edge_y = int(np.floor(self.RAW_CAM_W * a / 2 + b))  # height of edge's midpoint
        if ref_ratio is not None:
            self.ref_ratio = ref_ratio
            # self.edge_y = int(np.floor(self.edge_y * ref_ratio))
            # self.edge_line[0] = self.edge_line[0] * ref_ratio

        # print("HERE: "+ str(self.CamID) + ": " + str(self.ref_ratio))
        adjust_w = round(w * self.ref_ratio)
        adjust_h = round(h * self.ref_ratio)
        new_shape = (adjust_w, adjust_h)
        if self.ref_ratio > 1:
            rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
            ah, aw = rsz_image.shape[:2]
            dn = round(ah * 0.5 + h * 0.5)
            up = round(ah * 0.5 - h * 0.5)
            lt = round(aw * 0.5 - w * 0.5)
            rt = round(aw * 0.5 + w * 0.5)
            rsz_image = rsz_image[up:dn, lt:rt]
        else:
            rsz_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)

        self.img = rsz_image

