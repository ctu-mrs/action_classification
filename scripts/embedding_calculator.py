#!/usr/bin/env python3
import numpy as np
import math
from utils.landmark_feature_initializer import LandmarkFeatureInitializer


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
            "joint_pair_vectors",
            "joint_pair_vels",
            "joint_pair_accs",
            "joint_pair_vector_angle",
            "joint_pair_vector_angular_vels",
            "joint_pair_vector_angular_accs",
            "tri_joint_angles",
            "tri_joint_angular_vels",
            "tri_joint_angular_accs",
            "displacement_vectors",
        ]

    def __call__(self, normalized_landmarks, feature_list):
        assert normalized_landmarks.shape[0] == len(
            self._landmark_names
        ), "Unexpected number of landmarks: {}".format(normalized_landmarks.shape[0])
        # Get pose landmarks.
        landmarks = np.copy(normalized_landmarks)
        embedding = self._get_pose_embedding(landmarks)
        return embedding

    def _get_pose_embedding(self, landmarks):
        embedding = np.empty()
        return embedding

    def _get_joint_vectors(self, landmarks, joint_name):
        return landmarks[self._landmark_names.index(joint_name)]

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
