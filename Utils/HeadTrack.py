import cv2
import mediapipe as mp
import numpy as np

DEBUG = False


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)


class HeadTrack:

    def __init__(self):
        self.tilt_buffer = []
        self.center_y_offset = 0
        self.center_x_offset = 0
        self.calib = False

    def set_calib(self):
        self.calib = True

    def center_head(self, x_offset, y_offset):
        self.center_x_offset = -x_offset
        self.center_y_offset = -y_offset
        self.calib = False

    def HeadTacker(self, cam_feed, outputFrame, hist=20, fovRatio=1.78, sensitivity=6, h_bound=0.8):

        user_cam = cv2.cvtColor(cv2.flip(cam_feed, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        user_cam.flags.writeable = False

        # Get the result
        results = face_mesh.process(user_cam)

        # To improve performance
        user_cam.flags.writeable = True

        # Convert the color space from RGB to BGR
        user_cam = cv2.cvtColor(user_cam, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = user_cam.shape
        out_h, out_w, out_c = outputFrame.shape
        fov_h = round(out_h*h_bound)
        fov_w = round(fovRatio*fov_h)  # field of view in width take 90% of the full canvas to left moving space
        halfFOV_h = round(fov_h / 2)
        halfFOV_w = round(fov_w / 2)
        y_center = round(out_w / 2)
        x_center = round(out_h / 2)

        headbound_R = headbound_U = sensitivity
        headbound_L = headbound_D = -sensitivity

        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if self.calib:
                    self.center_head(x, y)

                x += self.center_x_offset
                y += self.center_y_offset
                head_dir = []

                if headbound_D <= x <= headbound_U:  # if head direction is in boundary
                    head_dir.append(x)
                elif x < headbound_D:  # if head direction is far down
                    head_dir.append(headbound_D)
                else:  # if head direction is far up
                    head_dir.append(headbound_U)

                if headbound_L <= y <= headbound_R:  # if head direction is in boundary
                    head_dir.append(y)
                elif y < headbound_L:  # if head direction is far left
                    head_dir.append(headbound_L)
                else:  # if head direction is far right
                    head_dir.append(headbound_R)

                if len(self.tilt_buffer) >= hist:
                    del self.tilt_buffer[0]  # pop the first element of the buffer when buffer is longer than limit
                self.tilt_buffer.append(head_dir)

                loc_tilt_buffer = self.tilt_buffer
                # print(len(loc_tilt_buffer))
                stb_tilt = np.mean(loc_tilt_buffer, axis=0)
                stb_x, stb_y = stb_tilt

                tilt_y_step = (y_center - halfFOV_w) / headbound_R  # calculate the ratio of head tilt to image rolling
                tilt_x_step = (x_center - halfFOV_h) / headbound_U  # calculate the ratio of head tilt to image rolling

                if headbound_D <= stb_x <= headbound_U:  # when head is in boundary
                    U = round(x_center - halfFOV_h - stb_x * tilt_x_step)
                    D = round(x_center + halfFOV_h - stb_x * tilt_x_step)
                elif stb_x < headbound_D:  # when head is too low
                    U = out_h-fov_h
                    D = out_h
                else:  # when head is too high
                    U = 0
                    D = fov_h

                if headbound_L <= stb_y <= headbound_R:  # when head is in boundary
                    L = round(y_center - halfFOV_w + stb_y * tilt_y_step)
                    R = round(y_center + halfFOV_w + stb_y * tilt_y_step)
                elif stb_y < headbound_L:  # when head is too left
                    L = 0
                    R = fov_w
                else:  # when head is too left
                    L = out_w-fov_w
                    R = out_w

                outputFrame = outputFrame[U:D, L:R]

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(user_cam, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(user_cam, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(user_cam, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(user_cam, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(
                image=user_cam,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        else:  # in case if face is not find, display the centered image
            left_boundary = y_center - halfFOV_w
            right_boundary = y_center + halfFOV_w
            up_boundary = x_center - halfFOV_h
            down_boundary = x_center + halfFOV_h
            outputFrame = outputFrame[up_boundary:down_boundary, left_boundary:right_boundary]

        # cv2.imshow("debug", user_cam)

        return outputFrame


if DEBUG:
    ht = HeadTrack()
    while cap.isOpened():
        success, image = cap.read()

        if success:
            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB

            img_processed = ht.HeadTacker(image, StaticImg, fovRatio=1, sensitivity=6)
            cv2.imshow("head", img_processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            ht.set_calib()

    cap.release()
