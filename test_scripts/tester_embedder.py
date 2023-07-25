import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

scriptpath = "../scripts"
sys.path.append(os.path.abspath(scriptpath))
from embedding_calculator import EmbeddingCalculator


class TestEmbeddingCalculator(unittest.TestCase):
    def setUp(self):
        # Create an instance of EmbeddingCalculator
        self.embedding_calculator = EmbeddingCalculator()

    def test_get_joint_vector(self):
        # Mock the LandmarkFeatureInitializer and its methods
        landmark_feature_initializer = MagicMock()
        landmark_feature_initializer.incoming_landmark_names = ["left_shoulder"]
        landmarks = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Sample landmarks
        self.embedding_calculator._all_single_joints[
            "single_lshoulder_object"
        ] = landmark_feature_initializer

        # Call the method being tested
        vector = self.embedding_calculator._get_joint_vector(
            landmarks, landmark_feature_initializer
        )

        # Assertions
        self.assertTrue(np.allclose(vector, np.array([1.0, 2.0, 3.0])))

    def test_get_joint_vel(self):
        # Mock the LandmarkFeatureInitializer and its methods
        landmark_feature_initializer = MagicMock()
        landmark_feature_initializer.incoming_landmark_names = ["left_shoulder"]
        landmarks = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Sample landmarks
        self.embedding_calculator._all_single_joints[
            "single_lshoulder_object"
        ] = landmark_feature_initializer

        # Set previous joint vector for velocity calculation
        landmark_feature_initializer.previous_joint_vector = np.array([0.0, 0.0, 0.0])

        # Call the method being tested
        vel = self.embedding_calculator._get_joint_vel(
            landmarks, landmark_feature_initializer
        )

        # Assertions
        self.assertTrue(np.allclose(vel, np.array([1.0, 2.0, 3.0])))

    def test_get_joint_acc(self):
        # Mock the LandmarkFeatureInitializer and its methods
        landmark_feature_initializer = MagicMock()
        landmark_feature_initializer.incoming_landmark_names = ["left_shoulder"]
        landmarks = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Sample landmarks
        self.embedding_calculator._all_single_joints[
            "single_lshoulder_object"
        ] = landmark_feature_initializer

        # Set previous joint velocity for acceleration calculation
        landmark_feature_initializer.previous_joint_vel = np.array([0.0, 0.0, 0.0])

        # Call the method being tested
        acc = self.embedding_calculator._get_joint_acc(
            landmarks, landmark_feature_initializer
        )

        # Assertions
        self.assertTrue(np.allclose(acc, np.array([1.0, 2.0, 3.0])))

    def test_call(self):
        # Mock the LandmarkFeatureInitializer and its methods for a few single joints
        single_joint_objects = {
            "single_lshoulder_object": MagicMock(),
            "single_rshoulder_object": MagicMock(),
        }
        for joint_name, joint_object in single_joint_objects.items():
            joint_object.incoming_landmark_names = [joint_name.split("_")[1]]

        landmarks = np.array(
            [[1.0, 2.0, 3.0]] * 13
        )  # Sample landmarks with 13 landmarks

        # Mock the get_pose_embedding method to return some dummy embedding
        self.embedding_calculator._get_pose_embedding = MagicMock(
            return_value=np.array([0.1, 0.2, 0.3])
        )

        # Set the single joints in the EmbeddingCalculator
        self.embedding_calculator._all_single_joints = single_joint_objects

        # Call the method being tested
        embedding = self.embedding_calculator(landmarks, time_stamp=1.0)

        # Assertions
        self.assertTrue(np.allclose(embedding, np.array([0.1, 0.2, 0.3])))


if __name__ == "__main__":
    unittest.main()
