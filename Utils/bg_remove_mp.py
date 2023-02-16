import mediapipe as mp
import cv2
import numpy as np


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
    Threshold = 0.5

    def __init__(self):
        self.background_mp = mp.solutions.selfie_segmentation
        self.mp_selfie_segmentation = self.background_mp.SelfieSegmentation
        self.selfie_segmentation = self.mp_selfie_segmentation(model_selection=0)

    def blur_bg(self, frame, threshold=Threshold, blur_intensity_x=51, blur_intensity_y=51):
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

    def replace_bg(self, frame, background, threshold=Threshold):
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

    def mask_bg(self, frame, threshold=Threshold):
        # todo not used consider deletion
        # param frame needs to be portrait
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_RGB.flags.writeable = False
        result = self.selfie_segmentation.process(frame_RGB)
        mask = result.segmentation_mask
        grescale_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        frame_RGB.flags.writeable = True
        condition = np.stack((mask,) * 3, axis=-1) > threshold
        return grescale_mask, condition


def stackIMG(cam_dict, bg_img, fit_shape, w_step, margins):
    loc_bgIMG = bg_img.copy()
    bg_h, bg_w, bg_c = bg_img.shape
    reference_y = int(np.floor(bg_h * 0.77))  # reference table line in background
    table_color = (64, 64, 64)
    loc_bgIMG[reference_y:bg_h, 0:bg_w] = table_color

    camCounter = 0

    if cam_dict:
        fit_h, fit_w = fit_shape
        h_margin, w_margin = margins[0], margins[1]
        for camID in cam_dict:
            frameClass = cam_dict[camID]  # extract the frame of current camera
            frame = frameClass.img
            frame_h, frame_w, frame_c = frame.shape
            edge_y = frameClass.edge_y  # extract the edge location on y-axis
            edge_a, edge_b = frameClass.edge_line  # extract the line equation parameters a, b
            ratio = fit_h / frame_h  # calculate the ratio of fit shape respect to original shape
            loc_edge_b = edge_b * ratio  # translate the edge height to current pixel coordinate
            loc_edge_y = edge_y * ratio

            background_y = round(loc_edge_y * bg_h / fit_h - h_margin*0.5)  # transfer the edge location to background coordinate
            shift_y = background_y - reference_y  # get the distance reference to the background edge line
            rsz_cam = cv2.resize(frame, (fit_w, fit_h))

            edge_left = (0, round(loc_edge_b))
            edge_right = (fit_w - 1, round(edge_a * fit_w + loc_edge_b))
            # if (loc_edge_b < fit_h) == (loc_edge_b < fit_h):
            # if both sides of the two edge lines are not crossing the boundary
            lower_left = (0, fit_h - 1)
            lower_right = (fit_w, fit_h - 1)
            contour = np.array([lower_left, lower_right, edge_right, edge_left])
            # elif loc_edge_b > fit_h:
            #     x_intercept = round(fit_h - loc_edge_b) / edge_a
            #     lower_left = (x_intercept, fit_h - 1)
            #     contour = np.array([lower_left, edge_left, edge_right])
            # else:
            #     x_intercept = round(fit_h - loc_edge_b) / edge_a
            #     lower_right = (x_intercept, fit_h - 1)
            #     contour = np.array([lower_right, edge_left, edge_right])

            lower_mask = np.zeros((fit_h, fit_w), np.uint8)
            cv2.drawContours(lower_mask, [contour], 0, 255, -1)

            # print("[DEBUG]:" + str(camID) + ": shift_y = " + str(shift_y))
            top_spacing = h_margin - shift_y
            bottom_spacing = h_margin + shift_y
            # print("[DEBUG]:" + str(camID) + ": top_spacing = " + str(top_spacing))
            x_left = w_step * camCounter + w_margin
            x_right = x_left + fit_w

            # image stacking and vertical alignment
            if top_spacing >= 0 and bottom_spacing >= 0:  # if the image is in between the boundary of bg
                cropped_cam = rsz_cam.copy()
                cropped_bg = loc_bgIMG[top_spacing:bg_h-bottom_spacing, x_left:x_right, :]

                cropped_lower_mask = lower_mask.copy()
                cropped_upper_mask = cv2.bitwise_not(cropped_lower_mask)

                u_l = (x_left, top_spacing)
                b_l = (x_left, bg_h-bottom_spacing)
                u_r = (x_right, top_spacing)
                b_r = (x_right, bg_h-bottom_spacing)
                fg_mask = np.zeros((bg_h, bg_w, bg_c), np.uint8)  # initialize the foreground mask w/ all black color
                cam_cnt = np.array([u_l, b_l, b_r, u_r])
                cam_cnt = scale_contour(cam_cnt, 0.87)  # scale down the contour to make the gradian change more natural
                cv2.drawContours(fg_mask, [cam_cnt], -1, (255, 255, 255), -1)  # mark foreground contour with white color
                fg_mask = cv2.GaussianBlur(fg_mask, (31, 31), 0)  # blur the edge of the foreground contour
                # cv2.imshow("mask" + str(camID), fg_mask)
                fg_mask = cv2.normalize(fg_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

                replacedBG_cam = segbg.replace_bg(cropped_cam, cropped_bg, threshold=.7)

                lower_cam = cv2.bitwise_and(cropped_cam, cropped_cam, mask=cropped_lower_mask)
                upper_cam = cv2.bitwise_and(replacedBG_cam, replacedBG_cam, mask=cropped_upper_mask)
                # color = unique_count_app(lower_cam)
                merged_cam = cv2.add(lower_cam, upper_cam)

                foreground = loc_bgIMG.copy()  # get a copy of current background for feathering
                background = loc_bgIMG.copy()
                foreground[top_spacing:bg_h-bottom_spacing, x_left:x_right, :] = merged_cam
                loc_bgIMG = background * (1 - fg_mask) + foreground * fg_mask
                loc_bgIMG = cv2.normalize(loc_bgIMG, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            elif top_spacing < 0:  # if image intercepts the upper boundary
                cropped_cam = rsz_cam[-top_spacing:fit_h, :, :]
                cropped_bg = loc_bgIMG[0:fit_h+top_spacing, x_left:x_right, :]
                cropped_lower_mask = lower_mask[-top_spacing:fit_h, :]
                cropped_upper_mask = cv2.bitwise_not(cropped_lower_mask)

                u_l = (x_left, 0)
                b_l = (x_left, fit_h+top_spacing)
                u_r = (x_right, 0)
                b_r = (x_right, fit_h+top_spacing)
                fg_mask = np.zeros((bg_h, bg_w, bg_c), np.uint8)  # initialize the foreground mask w/ all black color
                cam_cnt = np.array([u_l, b_l, b_r, u_r])
                cam_cnt = scale_contour(cam_cnt, 0.87)  # scale down the contour to make the gradian change more natural
                cv2.drawContours(fg_mask, [cam_cnt], -1, (255, 255, 255), -1)  # mark foreground contour with white color
                fg_mask = cv2.GaussianBlur(fg_mask, (31, 31), 0)  # blur the edge of the foreground contour
                # cv2.imshow("mask" + str(camID), fg_mask)
                fg_mask = cv2.normalize(fg_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
                # normalize the mask to the range of 0 to 1

                # mask, condition = segbg.mask_bg(cropped_cam)
                replacedBG_cam = segbg.replace_bg(cropped_cam, cropped_bg, threshold=.7)

                # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                lower_cam = cv2.bitwise_and(cropped_cam, cropped_cam, mask=cropped_lower_mask)
                upper_cam = cv2.bitwise_and(replacedBG_cam, replacedBG_cam, mask=cropped_upper_mask)
                # color = unique_count_app(lower_cam)
                merged_cam = cv2.add(lower_cam, upper_cam)

                foreground = loc_bgIMG.copy()  # get a copy of current background for feathering
                background = loc_bgIMG.copy()
                foreground[0:fit_h+top_spacing, x_left:x_right, :] = merged_cam
                loc_bgIMG = background * (1 - fg_mask) + foreground * fg_mask
                loc_bgIMG = cv2.normalize(loc_bgIMG, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # cv2.imshow("out" + str(camID), loc_bgIMG)
                # loc_bgIMG[-top_spacing:fit_h, x_left:x_right, :] = merged_cam

            else:  # if image intercepts the lower boundary
                cropped_cam = rsz_cam[0:fit_h+bottom_spacing, :, :]
                cropped_bg = loc_bgIMG[top_spacing:bg_h, x_left:x_right, :]
                cropped_lower_mask = lower_mask[0:fit_h+bottom_spacing, :]
                cropped_upper_mask = cv2.bitwise_not(cropped_lower_mask)

                u_l = (x_left, top_spacing)
                b_l = (x_left, bg_h)
                u_r = (x_right, top_spacing)
                b_r = (x_right, bg_h)
                fg_mask = np.zeros((bg_h, bg_w, bg_c), np.uint8)  # initialize the foreground mask w/ all black color
                cam_cnt = np.array([u_l, b_l, b_r, u_r])
                cam_cnt = scale_contour(cam_cnt, 0.87)  # scale down the contour to make the gradian change more natural
                cv2.drawContours(fg_mask, [cam_cnt], -1, (255, 255, 255), -1)  # mark foreground contour with white color
                fg_mask = cv2.GaussianBlur(fg_mask, (31, 31), 0)  # blur the edge of the foreground contour
                # cv2.imshow("mask" + str(camID), fg_mask)
                fg_mask = cv2.normalize(fg_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
                # normalize the mask to the range of 0 to 1

                replacedBG_cam = segbg.replace_bg(cropped_cam, cropped_bg, threshold=.7)
                lower_cam = cv2.bitwise_and(cropped_cam, cropped_cam, mask=cropped_lower_mask)
                upper_cam = cv2.bitwise_and(replacedBG_cam, replacedBG_cam, mask=cropped_upper_mask)
                merged_cam = cv2.add(lower_cam, upper_cam)

                foreground = loc_bgIMG.copy()  # get a copy of current background for blurring
                background = loc_bgIMG.copy()
                foreground[top_spacing:bg_h, x_left:x_right, :] = merged_cam
                loc_bgIMG = background * (1 - fg_mask) + foreground * fg_mask
                loc_bgIMG = cv2.normalize(loc_bgIMG, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # loc_bgIMG[0:fit_h - top_spacing, x_left:x_right, :] = merged_cam
            camCounter += 1

    return loc_bgIMG


def scale_contour(cnt, scale):
    # reference:
    # https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def unique_count_app(frame_w_blackmask):
    raw_frame_unfold = frame_w_blackmask.reshape(-1, frame_w_blackmask.shape[-1])
    non_black_mask = raw_frame_unfold != (0, 0, 0)
    non_black_mask = non_black_mask[:, 0] | non_black_mask[:, 1] | non_black_mask[:, 2]
    filtered_array = raw_frame_unfold[non_black_mask]
    if len(filtered_array) != 0:
        colors, count = np.unique(filtered_array, axis=0, return_counts=True)
        return colors[count.argmax()]
    else:
        black = (0, 0, 0)
        return black


segbg = SegmentationBG()

DEBUG = False

if DEBUG:
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 360)
    while cam.isOpened():
        success, raw_frame = cam.read()
        if success:
            raw_frame = cv2.resize(raw_frame, (640, 360))
            cv2.imshow("raw", raw_frame)
            main_color = unique_count_app(raw_frame)
            color_sample = np.full((100, 100, 3), main_color, dtype=np.uint8)
            cv2.imshow("color", color_sample)
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
