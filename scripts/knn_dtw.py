#!/usr/bin/env python3

import rospy
import roslib
import scipy.io as sio
import numpy as np
import os

from action_classification.msg import landmark3D
from feature_vector_generator import FeatureVectorEmbedder


class ROSInterfaceClass(object):
    def __init__(self):
        self.landmark_sub = rospy.Subscriber(
            "/landmark3Dcoords", landmark3D, self.landmarkCallback
        )
        self.landmark3D = None
        rospy.loginfo("Landmark Subscriber initialized")

    def landmarkCallback(self, msg):
        try:
            self.landmark3D = msg
        except Exception as e:
            rospy.logerr(e)

    def landmark3D_getter(self):
        return self.landmark3D


class PoseSample(object):
    def __init__(self, name, class_name, embedding):
        self.name = name
        self.class_name = class_name
        self.embedding = embedding

    def __repr__(self) -> str:
        return f"PoseSample(name={self.name}, class_name={self.class_name}, embedding={self.embedding})"


class ActionClassification(object):
    def __init__(
        self,
        embedding_dir,
        file_extension="mat",
        n_embeddings=341,
        n_dimensions=3,
        sliding_window_size=30,
        n_neighbors=10,
    ):
        self.n_embeddings = n_embeddings
        self.n_dimensions = n_dimensions
        self.sliding_window_size = sliding_window_size
        self.n_neighbors = n_neighbors

        self._embedding_samples = self._load_embedding_samples(
            embedding_dir, file_extension
        )

    def _load_embedding_samples(self, embedding_dir, file_extension):
        """
        Load .mat files by default. Each file encodes a pose sequence. The dimesions of the file are (n_embeddings, n_dimensions, n_frames).

        The folder structure is assumed to be:
        embedding_dir
        ├── class1
        │   ├── sample1.mat
        │   ├── sample2.mat
        │   └── ...
        ├── class2
        │   ├── sample1.mat
        │   ├── sample2.mat
        │   └── ...
        └── ...

        """

        embedding_samples = []
        for root, directories, files in os.walk(embedding_dir):
            for file in files:
                if file.endswith(file_extension):
                    mat_file_path = os.path.join(root, file)
                    data = sio.loadmat(mat_file_path)
                    embedding = data["embedding"]
                    # The class is the name of the sub folder that contains the .mat file
                    class_name = os.path.basename(root)
                    embedding_samples.append(PoseSample(file, class_name, embedding))
        return embedding_samples


def main():
    rospy.init_node("knn_dtw")
    rospy.loginfo("knn_dtw node initialized")

    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../embeddings_utd_mhad")

    # Load the embeddings
    action_classification = ActionClassification(embedding_path)
    print(action_classification._embedding_samples[0])


if __name__ == "__main__":
    main()
