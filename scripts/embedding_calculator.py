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
        # Single Joint Objects
        self.single_nose_object = LandmarkFeatureInitializer("nose")
        self.single_lshoulder_object = LandmarkFeatureInitializer("left_shoulder")
        self.single_rshoulder_object = LandmarkFeatureInitializer("right_shoulder")
        self.single_lelbow_object = LandmarkFeatureInitializer("left_elbow")
        self.single_relbow_object = LandmarkFeatureInitializer("right_elbow")
        self.single_lwrist_object = LandmarkFeatureInitializer("left_wrist")
        self.single_rwrist_object = LandmarkFeatureInitializer("right_wrist")
        self.single_lknee_object = LandmarkFeatureInitializer("left_knee")
        self.single_rknee_object = LandmarkFeatureInitializer("right_knee")
        self.single_lheel_object = LandmarkFeatureInitializer("left_heel")
        self.single_rheel_object = LandmarkFeatureInitializer("right_heel")

        # Joint Pairs
        # One Joint
        self.pair_lshoulder_lelbow_object = LandmarkFeatureInitializer(
            "left_shoulder", "left_elbow"
        )
        self.pair_rshoulder_relbow_object = LandmarkFeatureInitializer(
            "right_shoulder", "right_elbow"
        )
        self.pair_lelbow_lwrist_object = LandmarkFeatureInitializer(
            "left_elbow", "left_wrist"
        )
        self.pair_relbow_rwrist_object = LandmarkFeatureInitializer(
            "right_elbow", "right_wrist"
        )
        self.pair_lhip_lknee_object = LandmarkFeatureInitializer(
            "left_hip", "left_knee"
        )
        self.pair_rhip_rknee_object = LandmarkFeatureInitializer(
            "right_hip", "right_knee"
        )
        self.pair_lknee_lheel_object = LandmarkFeatureInitializer(
            "left_knee", "left_heel"
        )
        self.pair_rknee_rheel_object = LandmarkFeatureInitializer(
            "right_knee", "right_heel"
        )

        # Two Joints
        self.pair_lshoulder_lwrist_object = LandmarkFeatureInitializer(
            "left_shoulder", "left_wrist"
        )
        self.pair_rshoulder_rwrist_object = LandmarkFeatureInitializer(
            "right_shoulder", "right_wrist"
        )
        self.pair_lhip_lheel_object = LandmarkFeatureInitializer(
            "left_hip", "left_heel"
        )
        self.pair_rhip_rheel_object = LandmarkFeatureInitializer(
            "right_hip", "right_heel"
        )

        # Three Joints
        self.pair_lshoulder_lheel_object = LandmarkFeatureInitializer(
            "left_shoulder", "left_heel"
        )
        self.pair_rshoulder_rheel_object = LandmarkFeatureInitializer(
            "right_shoulder", "right_heel"
        )
        self.pair_lhip_lwrist_object = LandmarkFeatureInitializer(
            "left_hip", "left_wrist"
        )
        self.pair_rhip_rwrist_object = LandmarkFeatureInitializer(
            "right_hip", "right_wrist"
        )
        self.pair_lelbow_lknee_object = LandmarkFeatureInitializer(
            "left_elbow", "left_knee"
        )
        self.pair_relbow_rknee_object = LandmarkFeatureInitializer(
            "right_elbow", "right_knee"
        )

        # Cross Body
        self.pair_lelbow_relbow_object = LandmarkFeatureInitializer(
            "left_elbow", "right_elbow"
        )
        self.pair_lwrist_rwrist_object = LandmarkFeatureInitializer(
            "left_wrist", "right_wrist"
        )
        self.pair_lknee_rknee_object = LandmarkFeatureInitializer(
            "left_knee", "right_knee"
        )
        self.pair_lheel_rheel_object = LandmarkFeatureInitializer(
            "left_heel", "right_heel"
        )
        self.pair_lwrist_rheel_object = LandmarkFeatureInitializer(
            "left_wrist", "right_heel"
        )
        self.pair_lheel_rwrist_object = LandmarkFeatureInitializer(
            "left_heel", "right_wrist"
        )
        self.pair_lwrist_rknee_object = LandmarkFeatureInitializer(
            "left_wrist", "right_knee"
        )
        self.pair_lknee_rwrist_object = LandmarkFeatureInitializer(
            "left_knee", "right_wrist"
        )
        self.pair_lknee_rheel_object = LandmarkFeatureInitializer(
            "left_knee", "right_heel"
        )
        self.pair_lheel_rknee_object = LandmarkFeatureInitializer(
            "left_heel", "right_knee"
        )
        self.pair_lwrist_relbow_object = LandmarkFeatureInitializer(
            "left_wrist", "right_elbow"
        )
        self.pair_lelbow_rwrist_object = LandmarkFeatureInitializer(
            "left_elbow", "right_wrist"
        )
        self.pair_lknee_relbow_object = LandmarkFeatureInitializer(
            "left_knee", "right_elbow"
        )
        self.pair_lelbow_rknee_object = LandmarkFeatureInitializer(
            "left_elbow", "right_knee"
        )
        self.pair_lheel_relbow_object = LandmarkFeatureInitializer(
            "left_heel", "right_elbow"
        )
        self.pair_lelbow_rheel_object = LandmarkFeatureInitializer(
            "left_elbow", "right_heel"
        )
        self.pair_lshoulder_rknee_object = LandmarkFeatureInitializer(
            "left_shoulder", "right_knee"
        )
        self.pair_lknee_rshoulder_object = LandmarkFeatureInitializer(
            "left_knee", "right_shoulder"
        )
        self.pair_lshoulder_rheel_object = LandmarkFeatureInitializer(
            "left_shoulder", "right_heel"
        )
        self.pair_lheel_rshoulder_object = LandmarkFeatureInitializer(
            "left_heel", "right_shoulder"
        )
        self.pair_lhip_rwrists_object = LandmarkFeatureInitializer(
            "left_hip", "right_wrist"
        )
        self.pair_lwrists_rhip_object = LandmarkFeatureInitializer(
            "left_wrist", "right_hip"
        )

        # Tri Joints
        self.tri_lshoulder_lelbow_lwrist_object = LandmarkFeatureInitializer(
            "left_shoulder", "left_elbow", "left_wrist"
        )
        self.tri_rshoulder_relbow_rwrist_object = LandmarkFeatureInitializer(
            "right_shoulder", "right_elbow", "right_wrist"
        )
        self.tri_lhip_lknee_lheel_object = LandmarkFeatureInitializer(
            "left_hip", "left_knee", "left_heel"
        )
        self.tri_rhip_rknee_rheel_object = LandmarkFeatureInitializer(
            "right_hip", "right_knee", "right_heel"
        )
        self.tri_lshoulder_lhip_lknee_object = LandmarkFeatureInitializer(
            "left_shoulder", "left_hip", "left_knee"
        )
        self.tri_rshoulder_rhip_rknee_object = LandmarkFeatureInitializer(
            "right_shoulder", "right_hip", "right_knee"
        )
        self.tri_lelbow_lshoulder_lhip_object = LandmarkFeatureInitializer(
            "left_elbow", "left_shoulder", "left_hip"
        )
        self.tri_relbow_rshoulder_rhip_object = LandmarkFeatureInitializer(
            "right_elbow", "right_shoulder", "right_hip"
        )

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
