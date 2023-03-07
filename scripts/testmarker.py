#!/usr/bin/env python3

from pickle import NONE
import rospy
import roslib

from action_classification.msg import landmark, landmark3D
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker 

import numpy as np

class PoseVisualizer(object):
    def __init__(self):

        # _landmark_topic_ = rospy.get_param('~landmark_topic')
        # _landmark3D_topic_ = rospy.get_param('~landmark3D_topic')
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


        # self.landmark3D_sub_ = rospy.Subscriber(_landmark3D_topic_,\
        #                        landmark3D, self.landmark3DCallback)
        # self.landmark_sub_ = rospy.Subscriber(_landmark_topic_,\
        #                        landmark, self.landmarkCallback)
        
        self.marker_pub = rospy.Publisher("visualization_marker1", Marker,\
                                           queue_size = 2)
        # The second link is oriented at 90 degrees
        self.m = Marker()
        self.m.header.frame_id = "base"
        self.m.header.stamp = rospy.Time.now()
        self.m.ns = "poses"
        self.m.id = 1
        self.m.type = Marker.SPHERE_LIST
        self.m.action = Marker.ADD
        self.m.scale.x = 0.1
        self.m.scale.y = 0.1
        self.m.scale.z = 0.1
        # self.m.pose.position.x = 0
        # self.m.pose.position.y = 0
        # self.m.pose.position.z = 0
        # self.m.pose.orientation.x = 0
        # self.m.pose.orientation.y = 0
        # self.m.pose.orientation.z = 0
        self.m.pose.orientation.w = 1
        self.m.color.r = 0
        self.m.color.g = 1
        self.m.color.b = 1
        self.m.color.a = 1

    def sampleFunction(self):

        point1 = Point()
        point2 = Point()
        point1.x = 1
        point1.y = 1
        point1.z = 1
        point2.x = -1
        point2.y = -1
        point2.z = -1
        self.m.points.append(point1)
        self.m.points.append(point2)

        # self.m.points = (point1, point2)
        self.marker_pub.publish(self.m)


if __name__ == "__main__":
    rospy.init_node("gripper")   
    r = rospy.Rate(50)
    g = PoseVisualizer()
    while not rospy.is_shutdown():
        g.sampleFunction()
        r.sleep()
    rospy.spin()