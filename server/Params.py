
buff_4K = 4 * 1024

PORT = 9999
HOST_IP = '192.168.1.3'
MY_ADDR = (HOST_IP, PORT)

VID_W = 360
VID_H = 640
VID_SHAPE = (VID_H, VID_W, 3)


# original camera setting is in landscape (will be rotated to portrait view in cam thread)
RAW_CAM_W = VID_H

# so the camera's height & width is demo video's width & height
RAW_CAM_H = VID_W

RAW_CAM_SHAPE = (RAW_CAM_H, RAW_CAM_W, 3)

BG_W = 720
BG_H = 480
BG_SHAPE = (BG_H, BG_W, 3)
BG_DIM = (BG_W, BG_H)  # (w, h)

SHAPE = (RAW_CAM_H, RAW_CAM_W, 3)
R = RAW_CAM_H / RAW_CAM_W
# aspect ratio using the ratio of height to width to improve the efficiency of stackIMG function
BLUE = (255, 0, 0)