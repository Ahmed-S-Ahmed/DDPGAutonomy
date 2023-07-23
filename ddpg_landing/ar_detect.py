import cv2
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image
import cv_bridge
import threading


def get_features(img):
    image_center = np.array([img.shape[1]/2, img.shape[0]/2])
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
        parameters=arucoParams)
    if ids is not None:
        if [1] in ids:
            i, j = np.where(ids == 1)
            corners = corners[i[0]][0]
        else: 
            corners = corners[0][0]
        corner_0 = np.array(corners[0])
        corner_2 = np.array(corners[2])
        df = (corner_0 - corner_2) / 2
        marker_center = corner_0 - df 
        inframe = True
    else:
        inframe = False
        marker_center = np.array([0, 0])
        image_center = np.array([0, 0])
    err = image_center - marker_center
    return [err[0], err[1], inframe]