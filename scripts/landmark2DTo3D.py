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

        self.bridge_ = CvBridge()

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
    
        
    def depthExtractor(self, max_distance = 6):
        landmark2D_coords = self.landmark2D_coords_
        depth_image = self.depth_image_
        depth_cam_info = self.depth_cam_info_
        average_depth_array = self.avgDepthCalc(landmark2D_coords,\
                                    depth_cam_info, depth_image, max_distance)

    def normalizedToPixelCoordinates(self, coord_array, depth_cam_info):
        screen_size = (depth_cam_info.width, depth_cam_info.height)
        return tuple(round(coord * dimension) for coord, dimension in \
                                                zip(coord_array, screen_size))

    # Returns avg depth and depth of hip and shoulder points
    # (avgDepth, left_shoulder, right_shoulder, left_hip, right_hip)
    def avgDepthCalc(self, landmark2D_coords, depth_cam_info,\
                                            depth_image, max_distance):

        left_shoulder_depth_pixel = self.getPixelCoord("left_shoulder", 
            landmark2D_coords, depth_cam_info, depth_image, max_distance)
        right_shoulder_depth_pixel = self.getPixelCoord("right_shoulder", 
            landmark2D_coords, depth_cam_info, depth_image, max_distance)
        left_hip_depth_pixel = self.getPixelCoord("left_hip", 
            landmark2D_coords, depth_cam_info, depth_image, max_distance)
        right_hip_depth_pixel = self.getPixelCoord("right_hip", 
            landmark2D_coords, depth_cam_info, depth_image, max_distance)

        top_left_x = min(left_hip_depth_pixel[0], left_shoulder_depth_pixel[0],\
                    right_hip_depth_pixel[0], right_shoulder_depth_pixel[0])
        top_left_y = min(left_hip_depth_pixel[1], left_shoulder_depth_pixel[1],\
                    right_hip_depth_pixel[1], right_shoulder_depth_pixel[1])
        bot_right_x = max(left_hip_depth_pixel[0], left_shoulder_depth_pixel[0],\
                    right_hip_depth_pixel[0], right_shoulder_depth_pixel[0])
        bot_right_y = max(left_hip_depth_pixel[1], left_shoulder_depth_pixel[1],\
                    right_hip_depth_pixel[1], right_shoulder_depth_pixel[1])

        cv_image = self.bridge_.imgmsg_to_cv2(depth_image, depth_image.encoding)
        depth_rect_array = np.empty()

        for x_start in range(top_left_x, bot_right_x):
            for y_start in range(top_left_y, bot_right_y):
                np.append(depth_rect_array, cv_image[y_start, x_start])

        avg_person_dist = np.percentile(depth_rect_array, 25)

        
    def getPixelCoord(self, name, landmark2D_coords, depth_cam_info,\
                                            depth_image, max_distance):
        
        normalized_coord = \
            (landmark2D_coords.x[self.getLandmarkIndexByName(landmark2D_coords,\
                name)],\
            landmark2D_coords.y[self.getLandmarkIndexByName(landmark2D_coords,\
                name)])
        depth_pixel = self.normalizedToPixelCoordinates\
                            (normalized_coord, depth_cam_info)
        return depth_pixel

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