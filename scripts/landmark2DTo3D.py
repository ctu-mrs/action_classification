#!/usr/bin/env python3

from pickle import NONE
import rospy
import roslib

from mrs_msgs.msg import Float64Stamped
from std_msgs.msg import Float64, Int16
from sensor_msgs.msg import CompressedImage, Image
from action_classification.msg import landmark, landmark3D

import math
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


# Class to subscribe from the Depth topic and landmarks sent by mediapipe, 
# extract the depth coordinate from respective (x,y) supplied by mediapipe
# and add it to eh z coordinate of landmarks
class DepthInfoFromColour(object):
    def __init__(self):
        self.br = CvBridge()
        _colour_topic_ = rospy.get_param('~colour_topic')
        _uav_name_ = rospy.get_param('~uav_name')
        _aligned_depth_topic_ = rospy.get_param('aligned_depth_topic')
        _landmark_topic_ = rospy.get_param('~landmark_topic')

        self.landmark_sub_ = rospy.Subscriber(_landmark_topic_, landmark,\
                                                self.landmarkCallback)
        self.depth_sub_ = rospy.Subscriber(_aligned_depth_topic_, Image,\
                                                self.depthCallback)
        self.colour_sub_ = rospy.Subscriber(_colour_topic_, Image,\
                                                self.colourCallback)
        
        self.landmark3D_pub_ = rospy.Publisher(_uav_name_ +\
                        '/arthur/landnmark3Dcoords', landmark3D, queue_size=10)

        self.landmark3D_coords = landmark3D()
        self.landmark3D_coords