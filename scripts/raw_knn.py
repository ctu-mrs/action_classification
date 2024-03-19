import scipy.io as sio
import numpy as np
import os
from fastdtw import fastdtw
from sklearn.neighbors import BallTree
from feature_vector_generator import FeatureVectorEmbedder
from scipy.spatial.distance import euclidean
from sklearn.metrics import DistanceMetric
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model, Model
from multiprocessing import Pool
import time
import sys


currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from custom_classes import PoseSample, load_embedding_samples


class GestureClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.ball_tree = None
        self.labels = None

    def fit(self, X, y):
        # Calculate the DTW distance for each pair of samples
        dist = DistanceMetric.get_metric("pyfunc", func=self._dtw_distance)
        X = dist.pairwise(X)
        self.ball_tree = BallTree(X, metric="precomputed")
        self.labels = np.array(y)

    def predict(self, X):
        dist = DistanceMetric.get_metric("pyfunc", func=self._dtw_distance)
        X = dist.pairwise(X)
        dist, ind = self.ball_tree.query(X, k=self.n_neighbors)
        votes = self.labels[ind]
        return np.array([np.argmax(np.bincount(votes[i])) for i in range(len(votes))])

    @staticmethod
    def _dtw_distance(x, y):
        distance, _ = fastdtw(x, y, dist=euclidean)
        return distance
