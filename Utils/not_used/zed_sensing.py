import sys
import pyzed.sl as sl
import cv2
import numpy as np

# Create a ZED camera object
zed = sl.Camera()

# Set SVO path for playback
input_path = sys.argv[0]
init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file('vid/HD720_SN1542_16-49-42.svo')
init_parameters.camera_resolution = sl.RESOLUTION.HD720
init_parameters.camera_fps = 30
init_parameters.depth_mode = sl.DEPTH_MODE.QUALITY
init_parameters.coordinate_units = sl.UNIT.MILLIMETER
# init_parameters.depth_maximum_distance = 5000
init_parameters.depth_stabilization = False

runtime = sl.RuntimeParameters()
runtime.sensing_mode = sl.SENSING_MODE.FILL

# Open the ZED
zed = sl.Camera()
err = zed.open(init_parameters)

image_size = zed.get_camera_information().camera_resolution
w = zed.get_camera_information().camera_resolution.width
h = zed.get_camera_information().camera_resolution.height

# Create a sl.Mat with float type (32-bit)
svo_image = sl.Mat()
depth_zed = sl.Mat(w, h, sl.MAT_TYPE.F32_C1)
normal_map = sl.Mat(w, h, sl.MAT_TYPE.F32_C4)
# point_cloud = sl.Mat()
normal_disp = sl.Mat()

alpha = np.zeros([h, w], dtype=np.uint8)
# normal_BGR = np.zeros([h, w, 3], dtype=np.uint8)
BLACK = (0, 0, 0)

key = ' '
while key != 113:
    err = zed.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        # Read side by side frames stored in the SVO
        zed.retrieve_image(svo_image, sl.VIEW.LEFT, sl.MEM.CPU)
        zed.retrieve_image(depth_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)  # get depth
        zed.retrieve_image(normal_disp, sl.VIEW.NORMALS, sl.MEM.CPU, image_size)
        zed.retrieve_measure(normal_map, sl.MEASURE.NORMALS)  # get normal
        # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        norm_ocv = normal_disp.get_data()
        frame = svo_image.get_data()
        depth_ocv = depth_zed.get_data()
        normal_ocv = normal_map.get_data()
        # point_cloud_ocv = point_cloud.get_data()

        # mask_nan = (np.isnan(normal_ocv[:, :, 0]) & np.isnan(normal_ocv[:, :, 1]) & np.isnan(normal_ocv[:, :, 2]))

        # normal_BGR = normal_ocv[:, :, 0: 3].copy()
        # normal_BGR[~mask_nan] += 1
        # normal_BGR[~mask_nan] = (np.floor(normal_BGR[~mask_nan] * 127.5))
        # normal_BGR[mask_nan] = BLACK
        # normal_BGR = normal_BGR.astype(np.uint8)

        # alpha[mask_nan] = 0  # set nan to black
        # alpha[~mask_nan] = 255  # set nan to white
        cv2.imshow("norm", norm_ocv)
        cv2.imshow('video', frame)
        cv2.imshow('depth', depth_ocv)
        # Get frame count
        svo_position = zed.get_svo_position()
    elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        print("SVO end has been reached. Looping back to first frame")
        zed.set_svo_position(0)

    key = cv2.waitKey(1)

cv2.destroyAllWindows()
