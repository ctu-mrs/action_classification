#!/usr/bin/env python3

import rospy
import roslib
import scipy.io as sio
import numpy as np
import os
import fastdtw
from sklearn.neighbors import BallTree
from std_msgs.msg import String
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
        leaf_size=40,
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

        self._embedding_samples, self._class_names = self._load_embedding_samples(
            embedding_dir, file_extension
        )
        rospy.loginfo("Embedding samples loaded")
        self._ball_tree = self._generateBallTree(leaf_size=leaf_size)
        rospy.loginfo("Ball Tree Generated")

    def _generateBallTree(self, leaf_size=40):
        """
        This method generates a ball tree for the embedding samples. The ball tree is used to perform knn classification.
        """
        tree = BallTree(
            np.array(
                [
                    sample.embedding
                    for sample in self._embedding_samples
                    for i in range(self.n_embeddings - self.sliding_window_size + 1)
                ]
            ),
            leaf_size=leaf_size,
            metric=fastdtw,
        )
        return tree

    def knn_dtw_classify(self, embedding_seq, n_neighbors=10):
        """
        This methodes takes a sequence of embeddings and classify the sequence into a class as defined by the class names. It classifies the sequence using a ball tree implementation of knn and uses dtw as a distance metric.
        """

        # Embedding sequence is a 3D numpy array of shape (n_embeddings, n_dimensions, n_frames)
        # The embedding sequence is a sliding window of the original sequence
        # The sliding window is a 3D numpy array of shape (n_embeddings, n_dimensions, sliding_window_size)
        # The sliding window is used to compute the distance between the embedding sequence and the embedding samples
        # The distance is computed using dtw
        # The distance is computed between the embedding sequence and the embedding samples
        # The embedding samples are a list of PoseSample objects
        # The distance is computed between the embedding sequence and each embedding sample
        # The distance i computed using dtw between the embedding sequence and each embedding sample

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
        class_names = np.unique([sample.class_name for sample in embedding_samples])
        return embedding_samples, class_names


def main():
    rospy.init_node("knn_dtw")
    rospy.loginfo("knn_dtw node initialized")
    min_req_confidence = rospy.get_param("~min_req_confidence", 0.7)

    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../embeddings_utd_mhad")

    # Loading the embeddings
    rospy.loginfo("Loading the embeddings")
    # Load the embeddings
    action_classification = ActionClassification(embedding_path)
    # The ball tree will be created first in the action classification and then the subscribers will be initialized.

    # Get the 3D landmark coordinates
    rospy.loginfo("Initializing ROS interface")
    ros_interface = ROSInterfaceClass()
    rospy.loginfo("ROS interface initialized")

    # Initialize the feature vector embedder
    rospy.loginfo("Initializing the feature vector embedder")
    feature_vector_embedder = FeatureVectorEmbedder()
    rospy.loginfo("Feature vector embedder initialized")

    # Initialize the publisher
    rospy.loginfo("Initializing the publisher")
    pub = rospy.Publisher("/action_classification", String, queue_size=10)
    rospy.loginfo("Publisher initialized")

    # Initialize the rate
    rate = rospy.Rate(30)  # 30hz


if __name__ == "__main__":
    main()
