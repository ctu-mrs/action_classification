#!/usr/bin/env python3 
import numpy as np
import array
import math

class FeatureVectorEmbedder(object):
  def __init__(self):

    self._landmark_names = [
      'nose',
      'left_shoulder', 'right_shoulder',
      'left_elbow', 'right_elbow',
      'left_wrist', 'right_wrist',
      'left_hip', 'right_hip',
      'left_knee', 'right_knee',
      'left_heel', 'right_heel',
      ]                                                                                                  

  def __call__(self, landmarks):

    assert landmarks.shape[0] == len(self._landmark_names), \
            'Unexpected number of landmarks: {}'.format(landmarks.shape[0])
    # Get pose landmarks.
    landmarks = np.copy(landmarks)

    # Normalize landmarks.
    landmarks = self._normalize_pose_landmarks(landmarks)

    # Get embedding.
    feature_vector = self._get_feature_vector(landmarks)

    return feature_vector

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)

    # Normalize translation.
    pose_center = self._get_pose_center(landmarks)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 100

    return landmarks

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """Calculates pose size.
    
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # This approach uses only 2D landmarks to compute pose size.
    landmarks = landmarks[:, :2]

    # Hips center.
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # This approach assumes that the torso size provides a reasonable 
    # estimate of the overall body size or spatial extent of the pose. 
    # By adjusting the torso_size_multiplier, you can emphasize or de-emphasize
    # the influence of the torso size on the scaling factor.
    
    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)


    #This step helps prevent specific landmarks from dominating the scaling 
    # factor and ensures that the scaling takes into account the overall 
    # spatial extent of the pose, including potential outliers.

    # Max dist to pose center.
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)


        