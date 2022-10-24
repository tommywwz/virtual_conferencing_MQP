import cv2
import os
import numpy as np
import random

# !!!change with caution!!!
outputpath = "images/crop_temp"  # !!! this path will be cleared each time this program run!!!!
# !!!change with caution!!!

inputpath = "./images/raw/true"

img_name_cnt = 0

def segmentize(image_path):
    # Croping Formula ==> y:h, x:w
    global img_name_cnt
    idx, x_axis = 1, 0
    y_axis = 0
    img = cv2.imread(image_path)
    height, width, dept = img.shape
    segment_height = segment_width = y_height = x_width = int(width/4)
    remain_y = height
    remain_x = width
    while remain_y >= segment_height:
        while remain_x >= segment_width:
            crop = img[y_axis:y_height, x_axis:x_width]
            x_axis=x_width
            x_width+=segment_width
            cropped_image_path = "%s/%d.jpg" % (outputpath, img_name_cnt)
            img_name_cnt+=1
            cv2.imwrite(cropped_image_path, crop)
            remain_x = width - x_axis
            idx+=1
        remain_x = width
        y_axis += segment_height
        y_height += segment_height
        x_axis, x_width = 0, segment_width
        remain_y = height - y_axis


path_to_clear = outputpath
for f in os.listdir(path_to_clear):
    os.remove(os.path.join(path_to_clear, f))

for file in os.listdir(inputpath):
    segmentize("%s/%s" % (inputpath, file))




