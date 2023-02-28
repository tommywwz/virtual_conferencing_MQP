import cv2 as cv
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation

THRESHOLD_COLOR = 19.0


def stationary_bg_detection(bg_reference, curr_frame):
    diff1 = cv.subtract(bg_reference, curr_frame)
    diff2 = cv.subtract(curr_frame, bg_reference)
    diff = diff1 + diff2
    diff = np.floor(np.average(diff, axis=-1))
    mask_out = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
    mask_out[abs(diff) < THRESHOLD_COLOR] = 0
    mask_out[abs(diff) > THRESHOLD_COLOR] = 255

    kernel2x2 = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    kernel3x3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kernel_c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    cv.imshow("standard subtraction original", mask_out)
    mask_out = cv.erode(mask_out, kernel2x2, iterations=3)
    mask_out = cv.dilate(mask_out, kernel3x3, iterations=2)
    mask_out = cv.morphologyEx(mask_out, cv.MORPH_CLOSE, kernel_c)

    # h, w = mask_out.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)
    # frame_ff = mask_out.copy()
    # cv.floodFill(frame_ff, mask, (0, 0), 255)
    # frame_ff_inv = cv.bitwise_not(frame_ff)
    # mask_out = cv.bitwise_or(frame_ff_inv, mask_out)

    cv.imshow("standard subtraction out", mask_out)
    # cv.threshold(curr_h, 30, 255, cv.THRESH_BINARY, dst=curr_h)
    # cv.threshold(curr_h, 10, 255, cv.THRESH_BINARY, dst=curr_h)
    # cv.imshow("mask", curr_sub[:, :, 1])
    return mask_out


# backSub = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=600, detectShadows=False)
backSub = cv.createBackgroundSubtractorMOG2(history=100,
                                            varThreshold=2,
                                            detectShadows=False)  # history=1, varThreshold=100, detectShadows=False)

cam = cv.VideoCapture(1, cv.CAP_DSHOW)
cam.set(3, 640)  # width
cam.set(4, 360)  # height\
# cam.set(cv.CAP_PROP_EXPOSURE, -6)
# cam.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

success, frame = cam.read()
state_frame = frame
state_hsv = cv.cvtColor(state_frame, cv.COLOR_BGR2HSV)

segmentor = SelfiSegmentation()  # setup BGremover


while True:
    success, frame = cam.read()
    if success:
        # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = stationary_bg_detection(state_frame, frame)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        std_frame_out = frame.copy()
        std_frame_out[mask == 0] = 0
        cv.imshow("standard subtraction", std_frame_out)

        fgMask = backSub.apply(frame)

        fgMask[np.abs(fgMask) < 250] = 0

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        kernel_e = np.ones((2, 2), np.uint8)
        kernel_d = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))
        cv.imshow("FG mask", fgMask)
        fgMask = cv.erode(fgMask, kernel_e, iterations=2)  # reduce noise
        fgMask = cv.dilate(fgMask, kernel_d, iterations=2)  # augment important signals
        # opening = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel_d)
        closing = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)

        # contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(frame, contours, -1, color=(0, 255, 0), thickness=cv.FILLED)

        ml_sol = segmentor.removeBG(frame, (255, 0, 0), threshold=0.8)
        cv.imshow("ML solution", ml_sol)

        cv.imshow("original", frame)
        cv.imshow("MOG2 mask opening", fgMask)
        cv.imshow("MOG2 mask closing", closing)

        # lines = cv.HoughLinesP(closing, 1, np.pi / 180,
                               # threshold=15, lines=np.array([]), minLineLength=30, maxLineGap=3)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('f'):
            backSub = cv.createBackgroundSubtractorMOG2()  # history=1, varThreshold=100, detectShadows=False)
            state_frame = frame

        # if cv2.waitKey(1) & 0xFF == ord('f'):
        #     state_frame = frame
        #     state_grey = cv2.cvtColor(state_frame, cv2.COLOR_BGR2GRAY)
