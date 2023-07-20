#!/usr/bin/env python3

# Landmarks

# 0. nose                   17. left_pinky
# 1. left_eye_inner         18. right_pinky
# 2. left_eve               19. left_index
# 3. left_eye_outer         20. right_index
# 4. right_eye_inner        21. left_thumb
# 5. right_eye              22. right_thumb
# 6. right_eye_outer        23. left_hip
# 7. left_ear               24. right_hip
# 8. right_ear              25. left_knee
# 9. mouth_left             26. right_knee
# 10. mouth_right           27. left_ankle
# 11. left_shoulder         28. right_ankle
# 12. right_shoulder        29. left_heel
# 13. left_elbow            30. right_heel
# 14. right_elbow           31. left_foot index
# 15. left_wrist            32. right_foot_index
# 16. right_wrist

from pickle import NONE
import rospy
import roslib

from mrs_msgs.msg import ControlManagerDiagnostics
from mrs_msgs.msg import Float64Stamped
from mrs_msgs.msg import VelocityReferenceStamped
from mrs_msgs.msg import ReferenceStamped
from std_msgs.msg import Float64
from std_msgs.msg import Int16
from action_classification.msg import landmark
from feature_vector_generator import FeatureVectorEmbedder

import math
import array
import numpy as np
import time
import os
import sys

import cv2
from cv_bridge import CvBridge, CvBridgeError

import mediapipe as mp


# ROS Messages
from sensor_msgs.msg import CompressedImage, Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class PostTrackingClass(object):
    def __init__(self):
        self.br = CvBridge()
        self.image = NONE
        self.previous_sequence = -1  # To avoid repeat incoming message data
        self.total_frames = 0.0
        self.succ_frames = 0.0

        # Loading parameters from the launch file
        _camera_topic_ = rospy.get_param("~camera_topic")
        _uav_name_ = rospy.get_param("~uav_name")
        self._static_mode_ = rospy.get_param("~static_mode")
        self._model_complexity_ = rospy.get_param("~model_complexity")
        self._enable_segmentation_ = rospy.get_param("~enable_segmentation")
        self._min_detection_confidence_ = rospy.get_param("~min_detection_confidence")
        self._min_tracking_confidence_ = rospy.get_param("~min_tracking_confidence")

        self.landmarkpub = rospy.Publisher("landmarkCoord", landmark, queue_size=10)
        self.subscriber = rospy.Subscriber(_camera_topic_, Image, self.Callback)
        self.image_pub = rospy.Publisher("mediapipe/image_raw", Image, queue_size=10)

        # Initializing empty landmark-type variable
        self.landmarkcoords = landmark()
        self.landmarkcoords.x = array.array("f", (0 for f in range(0, 33)))
        self.landmarkcoords.y = array.array("f", (0 for f in range(0, 33)))
        self.landmarkcoords.vis = array.array("f", (0 for f in range(0, 33)))

        self._landmark_names = [
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_heel",
            "right_heel",
        ]

        rospy.loginfo("Pose Tracker Initialized")

    def Callback(self, ros_data, image_encoding="bgr8"):
        try:
            # Saving the image in a variable and convert it to cv2 compatible
            # format.
            self.image_header = ros_data.header
            self.image = self.br.imgmsg_to_cv2(ros_data, image_encoding)

        except CvBridgeError as e:
            if "[16UC1] is not a color format" in str(e):
                raise CvBridgeError(
                    "You may be trying to use a Image method "
                    + "(Subscriber, Publisher, conversion) on a depth image"
                    + " message. Original exception: "
                    + str(e)
                )
            raise e

    def PoseEstimator(self, pose):
        imagefiller = self.image
        current_image_time = self.image_header.stamp
        imageRGB = cv2.cvtColor(imagefiller, cv2.COLOR_BGR2RGB)
        results = pose.process(imageRGB)
        imageBGR = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        # If the current image seq was the same as the previous one,
        # then abort the current loop run
        if self.image_header.seq == self.previous_sequence:
            return
        self.previous_sequence = self.image_header.seq

        if results.pose_landmarks:
            i = 0
            self.succ_frames += 1.0
            to_be_sent_landmarks = ()
            # Saving each landmark's name and coordinate to our variable
            for landname in mp_pose.PoseLandmark:
                self.landmarkcoords.name.append(str(landname))

                self.landmarkcoords.x[i] = results.pose_landmarks.landmark[landname].x
                self.landmarkcoords.y[i] = results.pose_landmarks.landmark[landname].y
                self.landmarkcoords.vis[i] = results.pose_landmarks.landmark[
                    landname
                ].visibility
                if landname in self._landmark_names:
                    to_be_sent_landmarks += (
                        results.pose_landmarks.landmark[landname].x,
                        results.pose_landmarks.landmark[landname].y,
                        results.pose_landmarks.landmark[landname].visibility,
                    )

                i += 1

            self.landmarkcoords.header.frame_id = "Human"
            self.landmarkcoords.header.stamp = current_image_time

            mp_drawing.draw_landmarks(
                imageBGR,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            image_2BGR = cv2.cvtColor(imageBGR, cv2.COLOR_RGB2BGR)
            landnmark_img = self.br.cv2_to_imgmsg(image_2BGR, "rgb8")

            self.image_pub.publish(landnmark_img)
            self.landmarkpub.publish(self.landmarkcoords)
            embedder_obj = FeatureVectorEmbedder()
            embedding_to_be_displayed = embedder_obj(
                to_be_sent_landmarks, self.image_header.stamp.to_sec()
            )
            self.landmarkcoords.name = []

        # landnmark_img, imageRGB, imageBGR, image_2BGR = NONE


def main():
    rospy.init_node("Pose_Tracker", anonymous=True)
    rate = rospy.Rate(50)
    posetrackobject = PostTrackingClass()

    with mp_pose.Pose(
        # True for image input, False for video input
        static_image_mode=posetrackobject._static_mode_,
        model_complexity=posetrackobject._model_complexity_,  # 0, 1 or 2
        enable_segmentation=posetrackobject._enable_segmentation_,
        min_detection_confidence=posetrackobject._min_detection_confidence_,
        min_tracking_confidence=posetrackobject._min_tracking_confidence_,
    ) as pose:
        while not rospy.is_shutdown():
            if posetrackobject.image == NONE:
                continue

            rospy.loginfo_once("Starting Pose Tracking with MediaPipe Parameters: ")
            rospy.loginfo_once(
                "Static Image Mode: {}".format(posetrackobject._static_mode_)
            )
            rospy.loginfo_once(
                "Segmentation : {}".format(posetrackobject._enable_segmentation_)
            )
            rospy.loginfo_once(
                "Model Complexity: {}".format(posetrackobject._model_complexity_)
            )
            rospy.loginfo_once(
                "Minimun Detection Confidence: {}".format(
                    posetrackobject._min_detection_confidence_
                )
            )
            rospy.loginfo_once(
                "Minimum Tracking Conficence: {}".format(
                    posetrackobject._min_tracking_confidence_
                )
            )
            posetrackobject.total_frames += 1.0
            posetrackobject.PoseEstimator(pose)

            # To calculate the ratio of number of successfully detected frames
            # to the total number of frames streamed
            # print (landmarkObject.succ_frames / landmarkObject.total_frames)

            if cv2.waitKey(5) & 0xFF == 27:
                break
            rate.sleep()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down Landmark Detector Node")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
