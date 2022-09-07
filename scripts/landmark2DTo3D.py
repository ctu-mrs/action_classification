#!/usr/bin/env python3

from pickle import NONE
import rospy
import roslib

from mrs_msgs.msg import Float64Stamped
from std_msgs.msg import Float64, Int16
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from action_classification.msg import landmark, landmark3D

import math
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


# Class to subscribe from the Depth topic and landmarks sent by mediapipe, 
# extract the depth coordinate from respective (x,y) supplied by mediapipe
# and add it to the z coordinate of landmarks
class DepthInfoExtractor(object):
    def __init__(self):
        _uav_name_ = rospy.get_param('~uav_name')
        _aligned_depth_topic_ = rospy.get_param('aligned_depth_topic')
        _landmark_topic_ = rospy.get_param('~landmark_topic')
        _depth_cam_info_topic = rospy.get_param('_depth_cam_info_topic')

        self.landmark_sub_ = rospy.Subscriber(_landmark_topic_, landmark,\
                                                self.landmarkCallback)
        self.depth_sub_ = rospy.Subscriber(_aligned_depth_topic_, Image,\
                                                self.depthCallback)
        self.depth_cam_info_sub = rospy.Subscriber(_depth_cam_info_topic,\
                                        CameraInfo, self.depthCamInfoCallback)
        
        self.landmark3D_pub_ = rospy.Publisher('/landmark3Dcoords',\
                                                landmark3D, queue_size=10)

        self.landmark3D_coords_ = landmark3D()
        self.landmark2D_coords_ = landmark()

    def landmarkCallback(self, ros_data):
        self.landmark2D_coords_ = ros_data

    def depthCallback(self, ros_data):
        self.depth_image_ = ros_data 
    
    def depthCamInfoCallback(self, ros_data):
        self.depth_cam_info_ = ros_data
    
    def normalizedToPixelCoordinates(self, coord_array, depth_cam_info):
        screen_size = (depth_cam_info.width, depth_cam_info.height)
        return tuple(round(coord * dimension) for coord, dimension in \
                                                zip(coord_array, screen_size))
        

    def depthExtractor(self, max_distance = 6):
        landmark2D_coords = self.landmark2D_coords_
        depth_image = self.depth_image_
        depth_cam_info = self.depth_cam_info_
        left_shoulder_normalized_coord = (landmark2D_coords.x[self.getLandmarkIndexByName(landmark2D_coords,\
                "left_shoulder")], landmark2D_coords.y[self.getLandmarkIndexByName(landmark2D_coords,\
                "left_shoulder")])
        left_shoulder_depth_pixel = self.normalizedToPixelCoordinates\
                            (left_shoulder_normalized_coord, depth_cam_info)
        right_shoulder_normalized_coord = (landmark2D_coords.x[self.getLandmarkIndexByName(landmark2D_coords,\
                "right_shoulder")], landmark2D_coords.y[self.getLandmarkIndexByName(landmark2D_coords,\
                "right_shoulder")])
        right_shoulder_depth_pixel = self.normalizedToPixelCoordinates\
                            (left_shoulder_normalized_coord, depth_cam_info)
        left_hip_normalized_coord = (landmark2D_coords.x[self.getLandmarkIndexByName(landmark2D_coords,\
                "left_hip")], landmark2D_coords.y[self.getLandmarkIndexByName(landmark2D_coords,\
                "left_hip")])
        left_hip_depth_pixel = self.normalizedToPixelCoordinates\
                            (left_shoulder_normalized_coord, depth_cam_info)
        right_hip_normalized_coord = (landmark2D_coords.x[self.getLandmarkIndexByName(landmark2D_coords,\
                "right_hip")], landmark2D_coords.y[self.getLandmarkIndexByName(landmark2D_coords,\
                "right_hip")])
        right_hip_depth_pixel = self.normalizedToPixelCoordinates\
                            (left_shoulder_normalized_coord, depth_cam_info)
        avg_left_shoulder_depth = NONE

    def getLandmarkIndexByName(self, landmarks2D_or_3D, name):
        return np.where(landmarks2D_or_3D.name == name)[0][0]

         
 


def main():
    rospy.init_node('Depth_Info_Extractor', anonymous=True)
    depth_info_extractor_object = DepthInfoExtractor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down Depth Info Extractor")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()