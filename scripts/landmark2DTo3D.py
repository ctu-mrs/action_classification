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
        _aligned_depth_topic_ = rospy.get_param('~aligned_depth_topic')
        _landmark_topic_ = rospy.get_param('~landmark_topic')
        _depth_cam_info_topic = rospy.get_param('~depth_cam_info_topic')
        _max_depth_distance = rospy.get_param('~max_depth_distance')

        self.landmark_names_ = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
            ]



        self.bridge_ = CvBridge()

        self.landmark_sub_ = rospy.Subscriber(_landmark_topic_, landmark,\
                                                self.landmarkCallback)
        self.depth_sub_ = rospy.Subscriber(_aligned_depth_topic_, Image,\
                                                self.depthCallback)
        self.depth_cam_info_sub = rospy.Subscriber(_depth_cam_info_topic,\
                                        CameraInfo, self.depthCamInfoCallback)
        
        self.landmark3D_pub_ = rospy.Publisher('/landmark3Dcoords',\
                                                landmark3D, queue_size=10)

        self.landmark2D_coords_ = landmark()
        self.depth_image_ = Image()
        self.depth_image_ = None
        self.depth_cam_info_ = CameraInfo()
        self.depth_cam_info_ = None
        self.max_distance_ = _max_depth_distance

    def landmarkCallback(self, ros_data):
        self.landmark2D_coords_ = ros_data

    def depthCallback(self, ros_data):
        self.depth_image_ = ros_data
        self.encoding = ros_data.encoding 
        self.depthExtractor(self.depth_image_, 6)
    
    def depthCamInfoCallback(self, ros_data):
        self.depth_cam_info_ = ros_data
    
        
    def depthExtractor(self, depth_image, max_distance = 6):
        if self.depth_image_ ==None or self.landmark2D_coords_==None:
            print("Null Message")
            return
        landmark2D_coords = self.landmark2D_coords_
        depth_cam_info = self.depth_cam_info_
        landmark3D_to_send = landmark3D()
        landmark3D_to_send.header = landmark2D_coords
        landmark3D_to_send.name = landmark2D_coords.name
        landmark3D_to_send.vis = landmark2D_coords.vis
        landmark3D_to_send.x = landmark2D_coords.x
        landmark3D_to_send.y = landmark2D_coords.y
        
        average_depth = self.avgDepthCalc(landmark2D_coords,\
                                    depth_cam_info, depth_image, max_distance)

        if average_depth > self.max_distance_:
            average_depth = self.max_distance_
        
        # Hard Coded according to Realsense's FOV
        # 1.5mm per pixel at 1m distance => 60 pixels
        # 9mm per pixel at 6m distance => 10 pixels
        pixel_scale =  (int)(60 - (average_depth - 1)*10)
        if pixel_scale % 2 == 0:
            pixel_scale +=1
        
        for landmark_name_var in self.landmark_names_:
            pixel_coord_of_landmark = self.getPixelCoord(landmark_name_var, \
                landmark2D_coords, depth_cam_info, depth_image, max_distance)
            start_pixel_x = (int)(pixel_coord_of_landmark[0] -(pixel_scale-1)/2)
            start_pixel_y = (int)(pixel_coord_of_landmark[1] -(pixel_scale-1)/2)
            end_pixel_x = (int)(pixel_coord_of_landmark[0] +( pixel_scale-1)/2)
            end_pixel_y = (int)(pixel_coord_of_landmark[1] + (pixel_scale-1)/2)
            local_depth_array = np.array([])
            cv_image = self.bridge_.imgmsg_to_cv2(depth_image, self.encoding)
        
            for x_start in range (start_pixel_x, end_pixel_x):
                for y_start in range (start_pixel_y, end_pixel_y):
                    local_depth_array = \
                        np.append(local_depth_array, cv_image[y_start, x_start])
                    
            avg_landmark_depth = average_depth
            if local_depth_array.size >0:
                avg_landmark_depth = np.percentile(local_depth_array, 25)
            
            landmark_index = self.getLandmarkIndexByName(landmark_name_var)
            landmark3D_to_send.z[landmark_index] = avg_landmark_depth
            self.landmark3D_pub_.publish(landmark3D_to_send)
        

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

        cv_image = self.bridge_.imgmsg_to_cv2(depth_image, self.encoding)
        depth_rect_array = np.array([])

        for x_start in range(top_left_x, bot_right_x):
            for y_start in range(top_left_y, bot_right_y):
                depth_rect_array = \
                    np.append(depth_rect_array, cv_image[y_start, x_start])
        avg_person_dist = self.max_distance_        
        if depth_rect_array.size > 0:
            avg_person_dist = np.percentile(depth_rect_array, 25)

        return avg_person_dist

        
    def getPixelCoord(self, name, landmark2D_coords, depth_cam_info,\
                                            depth_image, max_distance):
        
        normalized_coord = \
            (landmark2D_coords.x[self.getLandmarkIndexByName(name)],\
            landmark2D_coords.y[self.getLandmarkIndexByName(name)])
        depth_pixel = self.normalizedToPixelCoordinates\
                            (normalized_coord, depth_cam_info)
        return depth_pixel

    def normalizedToPixelCoordinates(self, coord_array, depth_cam_info):
        screen_size = (depth_cam_info.width, depth_cam_info.height)
        return tuple(round(coord * dimension) for coord, dimension in \
                                                zip(coord_array, screen_size))

    def getLandmarkIndexByName(self, name):
        return np.where(np.array(self.landmark_names_) == name)[0][0]

         
 


def main():
    rospy.init_node('Depth_Info_Extractor', anonymous=True)
    rate = rospy.Rate(50)
    depth_info_extractor_object = DepthInfoExtractor()
    while not rospy.is_shutdown():
        rospy.loginfo_once("In While")
        # depth_info_extractor_object.depthExtractor()
        rate.sleep()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down Depth Info Extractor")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()