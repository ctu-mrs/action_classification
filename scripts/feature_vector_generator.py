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
        use_orientation_normalization=True,
        use_procrustes_normalization=True,
    ):
        assert landmarks.shape[0] == len(
            self._landmark_names
        ), "Unexpected number of landmarks: {}".format(landmarks.shape[0])
        # Get pose landmarks.
        landmarks = np.copy(landmarks)
        embedder = EmbeddingCalculator()
        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)
        if use_orientation_normalization == True:
            # This normalizes using the angle between shoulder and hip centres,
            # thus less effective in bent body positions.
            # rotated_landmarks = self._normalize_pose_orientation(landmarks)
            rotated_landmarks = self._normalize_pose_orientation_procrustes(landmarks)

            feature_vector = embedder(rotated_landmarks)
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
        left_shoulder = landmarks[self._landmark_names.index("left_shoulder")]
        right_shoulder = landmarks[self._landmark_names.index("right_shoulder")]
        left_hip = landmarks[self._landmark_names.index("left_hip")]
        right_hip = landmarks[self._landmark_names.index("right_hip")]

        # Calculate the vector representing the line connecting shoulder and hip centers
        shoulder_to_hip_vector = right_hip - left_shoulder

        # Calculate the angle of the shoulder-hip line
        angle_x = np.arctan2(shoulder_to_hip_vector[1], shoulder_to_hip_vector[2])
        angle_y = np.arctan2(shoulder_to_hip_vector[0], shoulder_to_hip_vector[2])
        angle_z = np.arctan2(shoulder_to_hip_vector[0], shoulder_to_hip_vector[1])

        # Apply the rotation to normalize the pose landmarks for orientation
        rotated_landmarks = np.copy(landmarks)
        cos_x = np.cos(-angle_x)
        sin_x = np.sin(-angle_x)
        cos_y = np.cos(-angle_y)
        sin_y = np.sin(-angle_y)
        cos_z = np.cos(-angle_z)
        sin_z = np.sin(-angle_z)
        for i in range(len(rotated_landmarks)):
            x = landmarks[i][0] - left_shoulder[0]
            y = landmarks[i][1] - left_shoulder[1]
            z = landmarks[i][2] - left_shoulder[2]
            rotated_landmarks[i][0] = (
                x * cos_y * cos_z
                - y * (cos_x * sin_z - sin_x * sin_y * cos_z)
                + z * (sin_x * sin_z + cos_x * sin_y * cos_z)
            )
            rotated_landmarks[i][1] = x * sin_y + y * cos_x * cos_y + z * sin_x * cos_y
            rotated_landmarks[i][2] = (
                -x * cos_y * sin_z
                + y * (cos_x * cos_z + sin_x * sin_y * sin_z)
                - z * (sin_x * cos_z - cos_x * sin_y * sin_z)
            )

        return rotated_landmarks

    def _normalize_pose_orientation_procrustes(self, landmarks):
        left_shoulder = landmarks[self._landmark_names.index("left_shoulder")]
        right_shoulder = landmarks[self._landmark_names.index("right_shoulder")]
        left_hip = landmarks[self._landmark_names.index("left_hip")]
        right_hip = landmarks[self._landmark_names.index("right_hip")]

        # Calculate the vector representing the line connecting shoulder and hip centers
        shoulder_to_hip_vector = right_hip - left_shoulder

        # Set the target direction for upright posture
        target_direction = np.array([0, 1, 0])  # [X, Y, Z] = [0, 1, 0] (upright)

        # Calculate the rotation quaternion to align the shoulder-hip line with the target direction
        rotation_quat = procrustes(shoulder_to_hip_vector, target_direction)[0]

        # Apply the rotation to normalize the pose landmarks for orientation
        rotated_landmarks = np.copy(landmarks)
        for i in range(len(rotated_landmarks)):
            rotated_landmarks[i] = rotation_quat.apply(rotated_landmarks[i])

        return rotated_landmarks
