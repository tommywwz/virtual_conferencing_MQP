import cv2
import numpy as np
import pixellib
from pixellib.tune_bg import alter_bg
from pixellib.instance import instance_segmentation

DEBUG = True


class BackgroundRemove:
    def __init__(self):
        self.change_bg = alter_bg()
        self.segment_image = instance_segmentation()
        self.segment_image.load_model("saved_model/mask_rcnn_coco.h5")

    def handle_frames(self, frame):
        # a function that remove the background of
        # for frame in frame_dict.values():
        #     segmask, output = self.segment_image.segmentImage(frame)
        return self.segment_image.segmentFrame(frame)


if DEBUG:
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 360)
    bg = BackgroundRemove()

    while cam.isOpened():
        success, raw_frame = cam.read()

        if success:

            raw_frame = cv2.resize(raw_frame, (640, 360))
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            segmask, output = bg.handle_frames(raw_frame)
            cv2.imshow("original", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.release()
                break

