UserCamID = 0
buff_4K = 4 * 1024

PORT = 9999
HOST_IP = '192.168.1.3'
MY_ADDR = (HOST_IP, PORT)

VID_W = 360
VID_H = 640
VID_SHAPE = (VID_H, VID_W, 3)

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

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