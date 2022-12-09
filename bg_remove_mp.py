import mediapipe as mp
import cv2
import numpy as np
DEBUG = False
#
# background_mp = mp.solutions.selfie_segmentation
# mp_selfie_segmentation = background_mp.SelfieSegmentation
# selfie_segmentation = mp_selfie_segmentation(model_selection=0)
# with mp_selfie_segmentation(model_selection=0) as selfie_segmentation:


# def BackgroundRemove(frame):
#     frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = bg_segmentation(.process())
#     return result


class SegmentationBG:
    def __init__(self):
        self.background_mp = mp.solutions.selfie_segmentation
        self.mp_selfie_segmentation = self.background_mp.SelfieSegmentation
        self.selfie_segmentation = self.mp_selfie_segmentation(model_selection=0)

    def blur_bg(self, frame, threshold=0.4, blur_intensity_x=51, blur_intensity_y=51):
        # param frame needs to be portrait
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_RGB.flags.writeable = False
        result = self.selfie_segmentation.process(frame_RGB)
        mask = result.segmentation_mask
        frame_RGB.flags.writeable = True
        frame_BGR = cv2.cvtColor(frame_RGB, cv2.COLOR_RGB2BGR)
        condition = np.stack((mask,) * 3, axis=-1) > threshold
        blurred = cv2.GaussianBlur(frame_BGR, (blur_intensity_x, blur_intensity_y), 0)

        output_image = np.where(condition, frame_BGR, blurred)
        out = cv2.normalize(output_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # Normalize the nparray to uint8 type
        return out

    def mask_bg(self, frame, threshold=0.4):
        # param frame needs to be portrait
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_RGB.flags.writeable = False
        result = self.selfie_segmentation.process(frame_RGB)
        mask = result.segmentation_mask
        frame_RGB.flags.writeable = True
        condition = np.stack((mask,) * 3, axis=-1) > threshold
        return mask, condition


if DEBUG:
    segbg = SegmentationBG()
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 360)
    while cam.isOpened():
        success, raw_frame = cam.read()
        if success:
            raw_frame = cv2.resize(raw_frame, (640, 360))
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            output = segbg.blur_bg(raw_frame)
            cv2.imshow("test", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.release()
                break
    # with mp_selfie_segmentation(model_selection=0) as selfie_segmentation:
    #     while cam.isOpened():
    #         success, raw_frame = cam.read()
    #         if success:
    #             raw_frame = cv2.resize(raw_frame, (640, 360))
    #             raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #
    #             frame_RGB = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    #             frame_RGB.flags.writeable = False
    #             result = selfie_segmentation.process(frame_RGB)
    #             frame_RGB.flags.writeable = True
    #             frame_BGR = cv2.cvtColor(frame_RGB, cv2.COLOR_RGB2BGR)
    #             mask = result.segmentation_mask
    #             a = np.stack((mask,) * 3, axis=-1)
    #             condition = np.stack((mask,) * 3, axis=-1) > 0.4
    #             # convert to three channels and compare if the value over the threshold
    #
    #             blurred = cv2.GaussianBlur(frame_BGR, (51, 51), 0)
    #
    #             output_image = np.where(condition, frame_BGR, blurred)
    #             out = cv2.normalize(output_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #             cv2.imshow("original", frame_BGR)
    #             cv2.imshow("blur", blurred)
    #             cv2.imshow("mask", mask)
    #             cv2.imshow("aaa", out)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 cam.release()
    #                 break
