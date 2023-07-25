#!/usr/bin/env python3
import numpy as np
import math
from scipy.spatial import procrustes
from embedding_calculator import EmbeddingCalculator


class FeatureVectorEmbedder(object):
    def __init__(self, torso_size_multiplier=2.5):
        self._torso_size_multiplier = torso_size_multiplier
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

    def __call__(
        self,
        landmarks,
        time_stamp,
        use_orientation_normalization=False,
    ):
        assert len(landmarks) == len(
            self._landmark_names
        ), "Unexpected number of landmarks: {}".format(len(landmarks[0]))
        # Get pose landmarks.
        landmarks = np.copy(landmarks)
        embedder = EmbeddingCalculator()
        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)
        if use_orientation_normalization == True:
            # This normalizes using the angle between shoulder and hip centres,
            # thus less effective in bent body positions.
            # rotated_landmarks = self._normalize_pose_orientation(landmarks)
            rotated_landmarks = self._normalize_pose_orientation(landmarks)

            feature_vector = embedder(rotated_landmarks, time_stamp)
            return feature_vector

        # Get embedding.
        feature_vector = embedder(landmarks, time_stamp)
        return feature_vector

    def _normalize_pose_landmarks(self, landmarks):
        # Normalizes landmarks translation, scale and orientation
        landmarks = np.copy(landmarks).astype("float64")

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it easier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index("left_hip")]
        right_hip = landmarks[self._landmark_names.index("right_hip")]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        landmarks = np.copy(landmarks)
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index("left_hip")]
        right_hip = landmarks[self._landmark_names.index("right_hip")]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index("left_shoulder")]
        right_shoulder = landmarks[self._landmark_names.index("right_shoulder")]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # This approach assumes that the torso size provides a reasonable
        # estimate of the overall body size or spatial extent of the pose.
        # By adjusting the torso_size_multiplier, you can emphasize or de-emphasize
        # the influence of the torso size on the scaling factor.

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # This step helps prevent specific landmarks from dominating the scaling
        # factor and ensures that the scaling takes into account the overall
        # spatial extent of the pose, including potential outliers.

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _normalize_pose_orientation(self, landmarks):
        rotated_landmarks = np.copy(landmarks)
        left_shoulder = rotated_landmarks[self._landmark_names.index("left_shoulder")]
        right_shoulder = rotated_landmarks[self._landmark_names.index("right_shoulder")]
        left_hip = rotated_landmarks[self._landmark_names.index("left_hip")]
        right_hip = rotated_landmarks[self._landmark_names.index("right_hip")]
        left_to_right_hip = right_hip - left_hip
        # Calculate the vector representing the line connecting shoulder and hip centers
        hip_to_shoulder_vector = (right_shoulder + left_shoulder) * 0.5 - (
            right_hip + left_hip
        ) * 0.5

        # Set the target direction for upright posture
        target_direction = np.array([1, 0, 0])  # [X, Y, Z] = [1, 0, 0] (upright)

        # Normalize the vector
        if np.linalg.norm(left_to_right_hip) != 0:
            left_to_right_hip /= np.linalg.norm(left_to_right_hip)
        else:
            left_to_right_hip = np.array([0, 0, 0])

        dot_product = np.dot(left_to_right_hip, target_direction)
        angle = math.acos(dot_product)
        if angle == 0:
            return rotated_landmarks
        cross_product = np.cross(left_to_right_hip, target_direction)
        # Rotate hip_to_shoulder_vector towards target_direction along the cross_product axis by angle
        rotation_matrix = self._rotation_matrix_from_axis_angle(cross_product, angle)
        rotated_landmarks = np.dot(rotation_matrix, rotated_landmarks.T).T
        print(rotated_landmarks)

        return rotated_landmarks

    def _rotation_matrix_from_axis_angle(self, axis, angle):
        """Creates rotation matrix corresponding to the rotation around given axis by angle."""
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(angle / 2.0)
        b, c, d = -axis * math.sin(angle / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = (
            b * c,
            a * d,
            a * c,
            a * b,
            b * d,
            c * d,
        )
        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
