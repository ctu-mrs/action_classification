#!/usr/bin/env python3
import numpy as np
import math
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from landmark_feature_initializer import LandmarkFeatureInitializer


# landmark structure
# Header header
#     uint32 seq
#     time stamp
#     string frame_id
# string[] name
# float32[] x
# float32[] y
# float32[] z
# float32[] vis
class EmbeddingCalculator(object):
    def __init__(self):
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

        self.feature_list_default = [
            "joint_vectors",
            "joint_vels",
            "joint_accs",
            "joint_vector_angles",
            "joint_angular_vels",
            "joint_angular_accs",
            "displacement_vectors",
            "joint_pair_vectors",
            "joint_pair_vels",
            "joint_pair_accs",
            "joint_pair_vector_angle",
            "joint_pair_vector_angular_vels",
            "joint_pair_vector_angular_accs",
            "tri_joint_angles",
            "tri_joint_angular_vels",
            "tri_joint_angular_accs",
        ]
        self._previous_time_stamp = 0.0
        self._current_time_stamp = 0.0
        self._all_single_joints = {
            # Single Joint Objects
            "single_nose_object": LandmarkFeatureInitializer("nose"),
            "single_lshoulder_object": LandmarkFeatureInitializer("left_shoulder"),
            "single_rshoulder_object": LandmarkFeatureInitializer("right_shoulder"),
            "single_lelbow_object": LandmarkFeatureInitializer("left_elbow"),
            "single_relbow_object": LandmarkFeatureInitializer("right_elbow"),
            "single_lwrist_object": LandmarkFeatureInitializer("left_wrist"),
            "single_rwrist_object": LandmarkFeatureInitializer("right_wrist"),
            "single_lknee_object": LandmarkFeatureInitializer("left_knee"),
            "single_rknee_object": LandmarkFeatureInitializer("right_knee"),
            "single_lheel_object": LandmarkFeatureInitializer("left_heel"),
            "single_rheel_object": LandmarkFeatureInitializer("right_heel"),
        }
        self._all_pair_joints = {
            # Joint Pairs
            ## One Joint
            "pair_lshoulder_lelbow_object": LandmarkFeatureInitializer(
                "left_shoulder", "left_elbow"
            ),
            "pair_rshoulder_relbow_object": LandmarkFeatureInitializer(
                "right_shoulder", "right_elbow"
            ),
            "pair_lelbow_lwrist_object": LandmarkFeatureInitializer(
                "left_elbow", "left_wrist"
            ),
            "pair_relbow_rwrist_object": LandmarkFeatureInitializer(
                "right_elbow", "right_wrist"
            ),
            "pair_lhip_lknee_object": LandmarkFeatureInitializer(
                "left_hip", "left_knee"
            ),
            "pair_rhip_rknee_object": LandmarkFeatureInitializer(
                "right_hip", "right_knee"
            ),
            "pair_lknee_lheel_object": LandmarkFeatureInitializer(
                "left_knee", "left_heel"
            ),
            "pair_rknee_rheel_object": LandmarkFeatureInitializer(
                "right_knee", "right_heel"
            ),
            ## Two Joints
            "pair_lshoulder_lwrist_object": LandmarkFeatureInitializer(
                "left_shoulder", "left_wrist"
            ),
            "pair_rshoulder_rwrist_object": LandmarkFeatureInitializer(
                "right_shoulder", "right_wrist"
            ),
            "pair_lhip_lheel_object": LandmarkFeatureInitializer(
                "left_hip", "left_heel"
            ),
            "pair_rhip_rheel_object": LandmarkFeatureInitializer(
                "right_hip", "right_heel"
            ),
            ## Three Joints
            "pair_lshoulder_lheel_object": LandmarkFeatureInitializer(
                "left_shoulder", "left_heel"
            ),
            "pair_rshoulder_rheel_object": LandmarkFeatureInitializer(
                "right_shoulder", "right_heel"
            ),
            "pair_lhip_lwrist_object": LandmarkFeatureInitializer(
                "left_hip", "left_wrist"
            ),
            "pair_rhip_rwrist_object": LandmarkFeatureInitializer(
                "right_hip", "right_wrist"
            ),
            "pair_lelbow_lknee_object": LandmarkFeatureInitializer(
                "left_elbow", "left_knee"
            ),
            "pair_relbow_rknee_object": LandmarkFeatureInitializer(
                "right_elbow", "right_knee"
            ),
            ## Cross Body
            "pair_lelbow_relbow_object": LandmarkFeatureInitializer(
                "left_elbow", "right_elbow"
            ),
            "pair_lwrist_rwrist_object": LandmarkFeatureInitializer(
                "left_wrist", "right_wrist"
            ),
            "pair_lknee_rknee_object": LandmarkFeatureInitializer(
                "left_knee", "right_knee"
            ),
            "pair_lheel_rheel_object": LandmarkFeatureInitializer(
                "left_heel", "right_heel"
            ),
            "pair_lwrist_rheel_object": LandmarkFeatureInitializer(
                "left_wrist", "right_heel"
            ),
            "pair_lheel_rwrist_object": LandmarkFeatureInitializer(
                "left_heel", "right_wrist"
            ),
            "pair_lwrist_rknee_object": LandmarkFeatureInitializer(
                "left_wrist", "right_knee"
            ),
            "pair_lknee_rwrist_object": LandmarkFeatureInitializer(
                "left_knee", "right_wrist"
            ),
            "pair_lknee_rheel_object": LandmarkFeatureInitializer(
                "left_knee", "right_heel"
            ),
            "pair_lheel_rknee_object": LandmarkFeatureInitializer(
                "left_heel", "right_knee"
            ),
            "pair_lwrist_relbow_object": LandmarkFeatureInitializer(
                "left_wrist", "right_elbow"
            ),
            "pair_lelbow_rwrist_object": LandmarkFeatureInitializer(
                "left_elbow", "right_wrist"
            ),
            "pair_lknee_relbow_object": LandmarkFeatureInitializer(
                "left_knee", "right_elbow"
            ),
            "pair_lelbow_rknee_object": LandmarkFeatureInitializer(
                "left_elbow", "right_knee"
            ),
            "pair_lheel_relbow_object": LandmarkFeatureInitializer(
                "left_heel", "right_elbow"
            ),
            "pair_lelbow_rheel_object": LandmarkFeatureInitializer(
                "left_elbow", "right_heel"
            ),
            "pair_lshoulder_rknee_object": LandmarkFeatureInitializer(
                "left_shoulder", "right_knee"
            ),
            "pair_lknee_rshoulder_object": LandmarkFeatureInitializer(
                "left_knee", "right_shoulder"
            ),
            "pair_lshoulder_rheel_object": LandmarkFeatureInitializer(
                "left_shoulder", "right_heel"
            ),
            "pair_lheel_rshoulder_object": LandmarkFeatureInitializer(
                "left_heel", "right_shoulder"
            ),
            "pair_lhip_rwrists_object": LandmarkFeatureInitializer(
                "left_hip", "right_wrist"
            ),
            "pair_lwrists_rhip_object": LandmarkFeatureInitializer(
                "left_wrist", "right_hip"
            ),
        }

        self._all_tri_joints = {
            # Tri Joints
            "tri_lshoulder_lelbow_lwrist_object": LandmarkFeatureInitializer(
                "left_shoulder", "left_elbow", "left_wrist"
            ),
            "tri_rshoulder_relbow_rwrist_object": LandmarkFeatureInitializer(
                "right_shoulder", "right_elbow", "right_wrist"
            ),
            "tri_lhip_lknee_lheel_object": LandmarkFeatureInitializer(
                "left_hip", "left_knee", "left_heel"
            ),
            "tri_rhip_rknee_rheel_object": LandmarkFeatureInitializer(
                "right_hip", "right_knee", "right_heel"
            ),
            "tri_lshoulder_lhip_lknee_object": LandmarkFeatureInitializer(
                "left_shoulder", "left_hip", "left_knee"
            ),
            "tri_rshoulder_rhip_rknee_object": LandmarkFeatureInitializer(
                "right_shoulder", "right_hip", "right_knee"
            ),
            "tri_lelbow_lshoulder_lhip_object": LandmarkFeatureInitializer(
                "left_elbow", "left_shoulder", "left_hip"
            ),
            "tri_relbow_rshoulder_rhip_object": LandmarkFeatureInitializer(
                "right_elbow", "right_shoulder", "right_hip"
            ),
        }

    def __call__(
        self,
        normalized_landmarks,
        time_stamp,
        feature_list=[
            "joint_vectors",
            "joint_vels",
            "joint_accs",
            "joint_vector_angles",
            "joint_angular_vels",
            "joint_angular_accs",
            "displacement_vectors",
            "joint_pair_vectors",
            "joint_pair_vels",
            "joint_pair_accs",
            "joint_pair_vector_angle",
            "joint_pair_vector_angular_vels",
            "joint_pair_vector_angular_accs",
            "tri_joint_angles",
            "tri_joint_angular_vels",
            "tri_joint_angular_accs",
        ],
    ):
        assert normalized_landmarks.shape[0] == len(
            self._landmark_names
        ), "Unexpected number of landmarks: {}".format(normalized_landmarks.shape[0])
        self._current_time_stamp = time_stamp
        self._feature_list = feature_list
        # Get pose landmarks.
        landmarks = np.copy(normalized_landmarks)
        embedding = self._get_pose_embedding(landmarks)
        return embedding

    def _get_pose_embedding(self, landmarks):
        all_single_joint_features = []

        for single_joint_object in self._all_single_joints.values():
            all_single_joint_features = (
                all_single_joint_features
                + self._perform_single_joint_operations(landmarks, single_joint_object)
            )

        all_pair_joint_features = []
        for pair_joint_object in self._all_pair_joints.values():
            all_pair_joint_features = (
                all_pair_joint_features
                + self._perform_joint_pair_operations(landmarks, pair_joint_object)
            )

        all_tri_joint_features = []
        for tri_joint_object in self._all_tri_joints.values():
            all_tri_joint_features = (
                all_tri_joint_features
                + self._perform_tri_joint_operations(landmarks, tri_joint_object)
            )

        embedding = np.array(
            all_single_joint_features + all_pair_joint_features + all_tri_joint_features
        )

        self.set_all_previous_variables()
        return embedding

    def _get_joint_vector(self, landmarks, single_joint_object):
        single_joint_object.joint_vector = landmarks[
            self._landmark_names.index(single_joint_object.incoming_landmark_names[0])
        ]
        return single_joint_object.joint_vector

    def _get_joint_vel(self, landmarks, single_joint_object):
        joint_vel = (
            single_joint_object.joint_vector - single_joint_object.previous_joint_vector
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return joint_vel

    def _get_joint_acc(self, landmarks, single_joint_object):
        joint_acc = (
            single_joint_object.joint_vel - single_joint_object.previous_joint_vel
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return joint_acc

    # A function that returns a vector of angles between a vector and x,y,z axis
    def _get_joint_vector_angle(self, landmarks, single_joint_object):
        single_joint_object.joint_vector_angle = np.array(
            [
                self._get_angle(single_joint_object.joint_vector, np.array([1, 0, 0])),
                self._get_angle(single_joint_object.joint_vector, np.array([0, 1, 0])),
                self._get_angle(single_joint_object.joint_vector, np.array([0, 0, 1])),
            ]
        )
        return single_joint_object.joint_vector_angle

    # A function that returns a vector of angular velocities between a vector and x,y,z axis
    def _get_joint_angular_vel(self, landmarks, single_joint_object):
        joint_angular_vel = (
            single_joint_object.joint_vector_angle
            - single_joint_object.previous_joint_vector_angle
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return joint_angular_vel

    # A function that returns a vector of angular accelerations between a vector and x,y,z axis
    def _get_joint_angular_acc(self, landmarks, single_joint_object):
        joint_angular_acc = (
            single_joint_object.joint_angular_vel
            - single_joint_object.previous_joint_angular_vel
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return joint_angular_acc

    # A function that returns displacepemt of a vector from its previous position
    def _get_displacement_vector(self, landmarks, single_joint_object):
        single_joint_object.displacement_vector = (
            single_joint_object.joint_vector - single_joint_object.previous_joint_vector
        )
        return single_joint_object.displacement_vector

    def _get_joint_pair_vector(self, landmarks, joint_pair_object):
        joint_pair_object.joint_pair_vector = self._get_distance_by_names(
            landmarks,
            joint_pair_object.incoming_landmark_names[0],
            joint_pair_object.incoming_landmark_names[1],
        )
        return joint_pair_object.joint_pair_vector

    def _get_joint_pair_vel(self, landmarks, joint_pair_object):
        joint_pair_vel = (
            joint_pair_object.joint_pair_vector
            - joint_pair_object.previous_joint_pair_vector
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return joint_pair_vel

    def _get_joint_pair_acc(self, landmarks, joint_pair_object):
        joint_pair_acc = (
            joint_pair_object.joint_pair_vel - joint_pair_object.previous_joint_pair_vel
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return joint_pair_acc

    def _get_joint_pair_vector_angle(self, landmarks, joint_pair_object):
        joint_pair_object.joint_pair_vector_angle = np.array(
            [
                self._get_angle(
                    joint_pair_object.joint_pair_vector, np.array([1, 0, 0])
                ),
                self._get_angle(
                    joint_pair_object.joint_pair_vector, np.array([0, 1, 0])
                ),
                self._get_angle(
                    joint_pair_object.joint_pair_vector, np.array([0, 0, 1])
                ),
            ]
        )
        return joint_pair_object.joint_pair_vector_angle

    def _get_joint_pair_vector_angular_vel(self, landmarks, joint_pair_object):
        joint_pair_angular_vel = (
            joint_pair_object.joint_pair_vector_angle
            - joint_pair_object.previous_joint_pair_vector_angle
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return joint_pair_angular_vel

    def _get_joint_pair_vector_angular_acc(self, landmarks, joint_pair_object):
        joint_pair_angular_acc = (
            joint_pair_object.joint_pair_angular_vel
            - joint_pair_object.previous_joint_pair_angular_vel
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return joint_pair_angular_acc

    def _get_tri_joint_angle(self, landmarks, tri_joint_object):
        # Takes three points a,b,c and returns the product of the unit cross product of vectors ab and bc and the angle between them.
        vector_ab = self._get_distance_by_names(
            landmarks,
            tri_joint_object.incoming_landmark_names[0],
            tri_joint_object.incoming_landmark_names[1],
        )
        vector_bc = self._get_distance_by_names(
            landmarks,
            tri_joint_object.incoming_landmark_names[1],
            tri_joint_object.incoming_landmark_names[2],
        )

        # Calculate the unit cross product of vectors ab and bc
        cross_product = np.cross(vector_ab, vector_bc)
        norm = np.linalg.norm(cross_product)
        if norm == 0:
            unit_cross_product = np.array([0, 0, 0])
        else:
            unit_cross_product = cross_product / norm

        # Calculate the angle between vectors ab and bc
        angle = self._get_angle(vector_ab, vector_bc)

        # Calculate the tri_joint_angle as the product of the unit cross product and the angle
        tri_joint_object.tri_joint_angle = angle * unit_cross_product

        return tri_joint_object.tri_joint_angle

    def _get_tri_joint_angular_vel(self, landmarks, tri_joint_object):
        tri_joint_angular_vel = (
            tri_joint_object.tri_joint_angle - tri_joint_object.previous_tri_joint_angle
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return tri_joint_angular_vel

    def _get_tri_joint_angular_acc(self, landmarks, tri_joint_object):
        tri_joint_angular_acc = (
            tri_joint_object.tri_joint_angular_vel
            - tri_joint_object.previous_tri_joint_angular_vel
        ) / (self._current_time_stamp - self._previous_time_stamp)
        return tri_joint_angular_acc

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from

    # A funtion that returns a vector of angles between two vectors
    def _get_angle(self, vector1, vector2):
        angle = np.arccos(
            np.dot(vector1, vector2)
            / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        )
        return angle

    def set_all_previous_variables(self):
        self._previous_time_stamp = self._current_time_stamp
        for single_joint_object in self._all_single_joints.values():
            single_joint_object.previous_joint_vector = np.copy(
                single_joint_object.joint_vector
            )

            single_joint_object.previous_joint_vel = np.copy(
                single_joint_object.joint_vel
            )
            single_joint_object.previous_joint_vector_angle = np.copy(
                single_joint_object.joint_vector_angle
            )

            single_joint_object.previous_joint_angular_vel = np.copy(
                single_joint_object.joint_angular_vel
            )

        for pair_joint_object in self._all_pair_joints.values():
            pair_joint_object.previous_joint_pair_vector = np.copy(
                pair_joint_object.joint_pair_vector
            )

            pair_joint_object.previous_joint_pair_vel = np.copy(
                pair_joint_object.joint_pair_vel
            )
            pair_joint_object.previous_joint_pair_vector_angle = np.copy(
                pair_joint_object.joint_pair_vector_angle
            )

            pair_joint_object.previous_joint_pair_angular_vel = np.copy(
                pair_joint_object.joint_pair_angular_vel
            )

        for tri_joint_object in self._all_tri_joints.values():
            tri_joint_object.previous_tri_joint_angle = np.copy(
                tri_joint_object.tri_joint_angle
            )

            tri_joint_object.previous_tri_joint_angular_vel = np.copy(
                tri_joint_object.tri_joint_angular_vel
            )

    def _perform_single_joint_operations(self, landmarks, single_joint_object):
        # Return a list of all the single joint operations
        single_joint_object.joint_vector = self._get_joint_vector(
            landmarks, single_joint_object
        )
        single_joint_object.joint_vel = self._get_joint_vel(
            landmarks, single_joint_object
        )
        single_joint_object.joint_acc = self._get_joint_acc(
            landmarks, single_joint_object
        )
        single_joint_object.joint_vector_angle = self._get_joint_vector_angle(
            landmarks, single_joint_object
        )
        single_joint_object.joint_angular_vel = self._get_joint_angular_vel(
            landmarks, single_joint_object
        )
        single_joint_object.joint_angular_acc = self._get_joint_angular_acc(
            landmarks, single_joint_object
        )
        single_joint_object.displacement_vector = self._get_displacement_vector(
            landmarks, single_joint_object
        )
        return [
            single_joint_object.joint_vector,
            single_joint_object.joint_vel,
            single_joint_object.joint_acc,
            single_joint_object.joint_vector_angle,
            single_joint_object.joint_angular_vel,
            single_joint_object.joint_angular_acc,
            single_joint_object.displacement_vector,
        ]

    def _perform_joint_pair_operations(self, landmarks, joint_pair_object):
        # Return a list of all the joint pair operations
        joint_pair_object.joint_pair_vector = self._get_joint_pair_vector(
            landmarks, joint_pair_object
        )
        joint_pair_object.joint_pair_vel = self._get_joint_pair_vel(
            landmarks, joint_pair_object
        )
        joint_pair_object.joint_pair_acc = self._get_joint_pair_acc(
            landmarks, joint_pair_object
        )
        joint_pair_object.joint_pair_vector_angle = self._get_joint_pair_vector_angle(
            landmarks, joint_pair_object
        )
        joint_pair_object.joint_pair_angular_vel = (
            self._get_joint_pair_vector_angular_vel(landmarks, joint_pair_object)
        )
        joint_pair_object.joint_pair_angular_acc = (
            self._get_joint_pair_vector_angular_acc(landmarks, joint_pair_object)
        )

        return [
            joint_pair_object.joint_pair_vector,
            joint_pair_object.joint_pair_vel,
            joint_pair_object.joint_pair_acc,
            joint_pair_object.joint_pair_vector_angle,
            joint_pair_object.joint_pair_angular_vel,
            joint_pair_object.joint_pair_angular_acc,
        ]

    def _perform_tri_joint_operations(self, landmarks, tri_joint_object):
        # Return a list of all the tri joint operations
        tri_joint_object.tri_joint_angle = self._get_tri_joint_angle(
            landmarks, tri_joint_object
        )
        tri_joint_object.tri_joint_angular_vel = self._get_tri_joint_angular_vel(
            landmarks, tri_joint_object
        )
        tri_joint_object.tri_joint_angular_acc = self._get_tri_joint_angular_acc(
            landmarks, tri_joint_object
        )

        return [
            tri_joint_object.tri_joint_angle,
            tri_joint_object.tri_joint_angular_vel,
            tri_joint_object.tri_joint_angular_acc,
        ]
