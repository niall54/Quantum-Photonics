# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:09:46 2020
3ecdc
@author: nmorone
"""

import cv2
import os

video_name = 'video_random.mp4'
fileprefix = '20210225_1610_00010'
images = [img for img in os.listdir() if img.endswith(".png") and img.startswith(fileprefix)]

frame = cv2.imread( images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 20, (width,height))

for image in images:
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()