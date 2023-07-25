#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import datetime
import unittest
from unittest.mock import MagicMock

scriptpath = "../scripts"
sys.path.append(os.path.abspath(scriptpath))
from embedding_calculator import EmbeddingCalculator

# Write an script that supplies the FeatureVector embedded with three sets of landmarks corresponding to time 1,2 and 3 second and test its output against the expected output.


class TestEmbeddingCalculator(object):
    def __init__(self):
        # Create an instance of EmbeddingCalculator
        self.embedding_calculator = EmbeddingCalculator()
        self.sample_landmarks1 = np.array(
            [
                [10.0, 10.0, 0.0],  # Nose
                [6, 9, 0.0],  # Left Shoulder
                [4, 9, 0.0],  # Right Shoulder
                [7, 7, 0.0],  # Left Elbow
                [3, 7, 0.0],  # Right Elbow
                [9, 4, 0.0],  # Left Wrist
                [1, 4, 0.0],  # Right Wrist
                [5.5, 5, 0.0],  # Left Hip
                [4.5, 5, 0.0],  # Right Hip
                [5.5, 3, 0.0],  # Left Knee
                [4.5, 3, 0.0],  # Right Knee
                [6, 0, 0.0],  # Left Heel
                [4, 0, 0.0],  # Right Heel
            ],
            dtype=np.float32,
        )

        self.sample_landmarks2 = np.array(
            [
                [10.0, 10.0, 0.0],  # Nose
                [6, 9, 0.0],  # Left Shoulder
                [4, 9, 0.0],  # Right Shoulder
                [7, 7, 0.5],  # Left Elbow
                [3, 7, 0.0],  # Right Elbow
                [9, 4, 1],  # Left Wrist
                [0, 5, 0.0],  # Right Wrist
                [5.5, 5, 0.0],  # Left Hip
                [4.5, 5, 0.0],  # Right Hip
                [5.5, 3, 0.0],  # Left Knee
                [4.5, 3, 0.0],  # Right Knee
                [6, 0, 0.0],  # Left Heel
                [4, 0, 0.0],  # Right Heel
            ],
            dtype=np.float32,
        )

        self.sample_landmarks3 = np.array(
            [
                [10.0, 10.0, 0.0],  # Nose
                [6, 9, 0.0],  # Left Shoulder
                [4, 9, 0.0],  # Right Shoulder
                [6, 9, 2],  # Left Elbow
                [0, 9, 0.0],  # Right Elbow
                [6, 9, 2],  # Left Wrist
                [-3, 9, 0.0],  # Right Wrist
                [5.5, 5, 0.0],  # Left Hip
                [4.5, 5, 0.0],  # Right Hip
                [5.5, 3, 0.0],  # Left Knee
                [4.5, 3, 0.0],  # Right Knee
                [6, 0, 0.0],  # Left Heel
                [4, 0, 0.0],  # Right Heel
            ],
            dtype=np.float32,
        )
        self.time_stamp = [0.5, 1.0, 1.5, 2.0, 2.5]

    def test_embedding_calculator(self):
        # Calculate the embedding for each set of landmarks
        self.embedding1 = self.embedding_calculator(
            self.sample_landmarks1, self.time_stamp[0]
        )
        print("embedding1 done")
        self.embedding2 = self.embedding_calculator(
            self.sample_landmarks2, self.time_stamp[1]
        )
        print("embedding2 done")
        self.embedding3 = self.embedding_calculator(
            self.sample_landmarks3, self.time_stamp[2]
        )
        print("embedding3 done")
        # self.embedding4 = self.embedding_calculator(
        #     self.sample_landmarks4, self.time_stamp[3]
        # )
        # self.embedding5 = self.embedding_calculator(
        #     self.sample_landmarks5, self.time_stamp[4]
        # )


def main():
    tester = TestEmbeddingCalculator()
    tester.test_embedding_calculator()

    i = 335
    j = 338
    print(tester.embedding1[i:j])
    print(tester.embedding2[i:j])
    print(tester.embedding3[i:j])


if __name__ == "__main__":
    main()
