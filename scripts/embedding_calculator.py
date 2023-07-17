#!/usr/bin/env python3
import numpy as np
import math


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

    def __call__(self, normalized_landmarks):
        assert normalized_landmarks.shape[0] == len(
            self._landmark_names
        ), "Unexpected number of landmarks: {}".format(normalized_landmarks.shape[0])
        # Get pose landmarks.
        landmarks = np.copy(normalized_landmarks)
        return
