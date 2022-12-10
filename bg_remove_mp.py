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

    def replace_bg(self, frame, background, threshold=0.4):
        # param frame needs to be portrait
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_RGB.flags.writeable = False
        result = self.selfie_segmentation.process(frame_RGB)
        mask = result.segmentation_mask
        frame_RGB.flags.writeable = True
        frame_BGR = cv2.cvtColor(frame_RGB, cv2.COLOR_RGB2BGR)
        condition = np.stack((mask,) * 3, axis=-1) > threshold

        output_image = np.where(condition, frame_BGR, background)
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


def stackIMG(cam_dict, bg_img, fit_shape, w_step, margins, cam_edge_y, edge_lines):
    loc_bgIMG = bg_img.copy()
    fit_h, fit_w = fit_shape
    bg_h, bg_w, bg_c = bg_img.shape
    reference_y = np.floor(bg_h * 6 / 7)  # reference table line in background
    h_margin, w_margin = margins[0], margins[1]
    i = 0

    for camID in cam_dict:
        frame = cam_dict[camID]  # extract the frame of current camera
        frame_h, frame_w, frame_c = frame.shape
        edge_y = cam_edge_y[camID]  # extract the edge location on y-axis

        background_y = np.floor(edge_y * fit_h / frame_h)  # transfer the edge location to background coordinate
        shift_y = int(background_y - reference_y)  # get the distance reference to the background edge line

        edge_a, edge_b = edge_lines[camID]  # extract the line equation parameters a, b

        ratio = fit_h / frame_h  # calculate the ratio of fit shape respect to original shape

        rsz_cam = cv2.resize(frame, (fit_w, fit_h))

        loc_edge_b = edge_b.copy() * ratio

        edge_left = (0, round(loc_edge_b))
        edge_right = (fit_w-1, round(edge_a*fit_w+loc_edge_b))
        if (loc_edge_b < fit_h) == (loc_edge_b < fit_h):
            # if both sides of the two edge lines are not crossing the boundary
            lower_left = (0, fit_h-1)
            lower_right = (fit_w, fit_h-1)
            contour = np.array([lower_left, lower_right, edge_right, edge_left])
        elif loc_edge_b > fit_h:
            x_intercept = round(fit_h-loc_edge_b)/edge_a
            lower_left = (x_intercept, fit_h-1)
            contour = np.array([lower_left, edge_left, edge_right])
        else:
            x_intercept = round(fit_h - loc_edge_b)/edge_a
            lower_right = (x_intercept, fit_h-1)
            contour = np.array([lower_right, edge_left, edge_right])

        lower_mask = np.zeros((fit_h, fit_w), np.uint8)
        cv2.drawContours(lower_mask, [contour], 0, 255, -1)

        print(str(camID)+": shift_y = "+str(shift_y))
        top_spacing = h_margin + shift_y
        x_left = w_step * i + w_margin
        x_right = x_left + fit_w

        if top_spacing <= 0:
            cropped_cam = rsz_cam[0:fit_h+top_spacing, :, :]
            cropped_bg = loc_bgIMG[-top_spacing:fit_h, x_left:x_right, :]

            cropped_lower_mask = lower_mask[0:fit_h+top_spacing, :]
            cropped_upper_mask = cv2.bitwise_not(cropped_lower_mask)

            blurredBG_cam = segbg.blur_bg(cropped_cam)
            replacedBG_cam = segbg.replace_bg(cropped_cam, cropped_bg)

            lower_cam = cv2.bitwise_and(blurredBG_cam, blurredBG_cam, mask=cropped_lower_mask)
            upper_cam = cv2.bitwise_and(replacedBG_cam, replacedBG_cam, mask=cropped_upper_mask)

            merged_cam = cv2.add(lower_cam, upper_cam)

            loc_bgIMG[-top_spacing:fit_h, x_left:x_right, :] = merged_cam
        else:
            cropped_cam = rsz_cam[top_spacing:fit_h, :, :]
            cropped_bg = loc_bgIMG[0:fit_h-top_spacing, x_left:x_right, :]

            cropped_lower_mask = lower_mask[top_spacing:fit_h, :]
            cropped_upper_mask = cv2.bitwise_not(cropped_lower_mask)

            blurredBG_cam = segbg.blur_bg(cropped_cam)
            replacedBG_cam = segbg.replace_bg(cropped_cam, cropped_bg)

            lower_cam = cv2.bitwise_and(blurredBG_cam, blurredBG_cam, mask=cropped_lower_mask)
            upper_cam = cv2.bitwise_and(replacedBG_cam, replacedBG_cam, mask=cropped_upper_mask)

            merged_cam = cv2.add(lower_cam, upper_cam)

            loc_bgIMG[0:fit_h - top_spacing, x_left:x_right, :] = merged_cam
        i += 1

    return loc_bgIMG


segbg = SegmentationBG()

if DEBUG:
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
