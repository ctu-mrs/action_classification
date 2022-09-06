#!/usr/bin/env python3
import numpy as np
import array
import math

class Full3DBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier

    # Names of the landmarks as they appear in the prediction.
    self._landmark_names = [
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

  def __call__(self, landmarks):
    """Normalizes pose landmarks and converts to embedding
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).
    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances defined in `_get_pose_distance_embedding`.
    """
    assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

    # Get pose landmarks.
    landmarks = np.copy(landmarks)

    # Normalize landmarks.
    landmarks = self._normalize_pose_landmarks(landmarks)

    # Get embedding.
    embedding = self._get_pose_distance_embedding(landmarks)

    return embedding

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

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)

    # Max dist to pose center.
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)

  def _get_pose_distance_embedding(self, landmarks):
    """Converts pose landmarks into 3D embedding.
    We use several pairwise 3D distances to form pose embedding. All distances
    include X and Y components with sign. We differnt types of pairs to cover
    different pose classes. Feel free to remove some or add new.
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).
    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances.
    """
    embedding = np.array([
      # One Joint
        self._get_distance(
          self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
          self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

        self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

        self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),



        # Two joints.

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

        # Three joints.

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),
        
        self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        # Cross body.

        self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
        self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

        self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
        self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

        # 3D Angles
        self._get_angle_embedding_product_by_names(landmarks, 'left_hip', \
                                              'left_shoulder', 'left_elbow'),
        self._get_angle_embedding_product_by_names(landmarks, 'right_hip', \
                                              'right_shoulder', 'right_elbow'),
        
        self._get_angle_embedding_product_by_names(landmarks, 'left_shoulder', \
                                              'left_hip', 'left_knee'),
        self._get_angle_embedding_product_by_names(landmarks, 'right_shoulder', \
                                              'right_hip', 'right_knee'),

        self._get_angle_embedding_product_by_names(landmarks, 'left_shoulder',\
                                              'left_elbow', 'left_wrist'),
        self._get_angle_embedding_product_by_names(landmarks, 'right_shoulder',\
                                              'right_elbow', 'right_wrist'),
                                    
        self._get_angle_embedding_product_by_names(landmarks, 'left_heel', \
                                                    'left_knee', 'left_hip'),
        self._get_angle_embedding_product_by_names(landmarks, 'right_heel', \
                                                    'right_knee', 'right_hip'),

        self._get_angle_embedding_product( landmarks, 
          landmarks[self._landmark_names.index('left_knee')],
          self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
          landmarks[self._landmark_names.index('right_knee')]
        ),
        self._get_angle_embedding_product( landmarks, 
          landmarks[self._landmark_names.index('left_elbow')],
          self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder'),
          landmarks[self._landmark_names.index('right_elbow')]
        ),
        
    ])

    return embedding

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

  def _get_angle_by_names(self, landmarks, startname, midname, endname):
    start_point = landmarks[self._landmark_names.index(startname)]
    mid_point = landmarks[self._landmark_names.index(midname)]
    end_point = landmarks[self._landmark_names.index(endname)]
    return self._get_angle(landmarks, start_point, mid_point, end_point)

  def _get_normal_by_names(self, landmarks, startname, midname, endname):
    start_point = landmarks[self._landmark_names.index(startname)]
    mid_point = landmarks[self._landmark_names.index(midname)]
    end_point = landmarks[self._landmark_names.index(endname)]
    return self._get_normal(landmarks, start_point, mid_point, end_point)

  def _get_angle(self, landmarks, start_point, mid_point, end_point):
    line1 = start_point - mid_point
    line2 = end_point - mid_point
    cosine_angle = np.dot(line1 , line2) / (np.linalg.norm(line1) \
                                                    * np.linalg.norm(line2))
    angle = np.arccos(cosine_angle)
    return angle

  def _get_normal(self, landmarks, start_point, mid_point, end_point):
    line1 = start_point - mid_point
    line2 = end_point - mid_point
    cross_prod = np.cross(line1, line2)
    norm = np.linalg.norm(cross_prod)
    if norm == 0:
      return norm
    return cross_prod / norm

  def _get_angle_embedding_product_by_names(self, landmarks, startname, midname, endname):
    angle = self._get_angle_by_names(landmarks, startname, midname, endname) 
    unit_normal = self._get_normal_by_names(landmarks, startname, midname, endname)
    # A metric which encodes the angle between the two vectors as the cosine, 
    # and the angle of the plane as the normal vector
    return math.cos(angle) * unit_normal

  def _get_angle_embedding_product(self, landmarks, start_point,\
                                               mid_point, end_point):
    angle = self._get_angle(landmarks, start_point, mid_point, end_point) 
    unit_normal = self._get_normal(landmarks, start_point, mid_point, end_point)
    # A metric which encodes the angle between the two vectors as the cosine, 
    # and the angle of the plane as the normal vector
    return math.cos(angle) * unit_normal

def main():
    pose_embedder = Full3DBodyPoseEmbedder()


if __name__ == '__main__':
    main()
