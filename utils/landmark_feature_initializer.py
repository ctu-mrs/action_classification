#!/usr/bin/env python3
import numpy as np
import math


class LandmarkFeatureInitializer(object):
    def __init__(self, *landmark_names):
        self.incoming_landmark_names = landmark_names
        _number_of_landmarks = len(self.incoming_landmark_names)

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
        for item in self.incoming_landmark_names:
            if item not in self._landmark_names:
                raise ValueError(
                    "Incorrect landmark name arguments to the Landmark Feature Initializer"
                )
        if _number_of_landmarks == 0:
            raise ValueError("Not Enough Arguments to the Landmark Feature Initializer")

        elif _number_of_landmarks == 1:
            self.previous_joint_vector = np.array([0, 0, 0], dtype=np.float32)
            self.previous_joint_vel = np.array([0, 0, 0], dtype=np.float32)
            self.previous_joint_vector_angle = np.array([0, 0, 0], dtype=np.float32)
            self.previous_joint_angular_vel = np.array([0, 0, 0], dtype=np.float32)

            self.joint_vector = np.array([0, 0, 0], dtype=np.float32)
            self.joint_vel = np.array([0, 0, 0], dtype=np.float32)
            self.joint_acc = np.array([0, 0, 0], dtype=np.float32)
            self.joint_vector_angle = np.array([0, 0, 0], dtype=np.float32)
            self.joint_angular_vel = np.array([0, 0, 0], dtype=np.float32)
            self.joint_angular_acc = np.array([0, 0, 0], dtype=np.float32)
            self.displacement_vector = np.array([0, 0, 0], dtype=np.float32)

        elif _number_of_landmarks == 2:
            self.previous_joint_pair_vector = np.array([0, 0, 0], dtype=np.float32)
            self.previous_joint_pair_vel = np.array([0, 0, 0], dtype=np.float32)
            self.previous_joint_pair_vector_angle = np.array(
                [0, 0, 0], dtype=np.float32
            )
            self.previous_joint_pair_angular_vel = np.array([0, 0, 0], dtype=np.float32)

            self.joint_pair_vector = np.array([0, 0, 0], dtype=np.float32)
            self.joint_pair_vel = np.array([0, 0, 0], dtype=np.float32)
            self.joint_pair_acc = np.array([0, 0, 0], dtype=np.float32)
            self.joint_pair_vector_angle = np.array([0, 0, 0], dtype=np.float32)
            self.joint_pair_angular_vel = np.array([0, 0, 0], dtype=np.float32)
            self.joint_pair_angular_acc = np.array([0, 0, 0], dtype=np.float32)

        elif _number_of_landmarks == 3:
            self.previous_tri_joint_angle = np.array([0, 0, 0], dtype=np.float32)
            self.previous_tri_joint_angular_vel = np.array([0, 0, 0], dtype=np.float32)

            self.tri_joint_angle = np.array([0, 0, 0], dtype=np.float32)
            self.tri_joint_angular_vel = np.array([0, 0, 0], dtype=np.float32)
            self.tri_joint_angular_acc = np.array([0, 0, 0], dtype=np.float32)

        else:
            raise ValueError("Too Many Arguments to the Landmark Feature Initializer")
