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
import time
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from custom_classes import PoseSample, load_embedding_samples


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
        self._embedding_samples, self._class_names = load_embedding_samples(
            embedding_dir=embedding_dir, file_extension=file_extension
        )

    def dtw_distances(self, X, Y):
        """
        This method takes in two embeddings, flattens the (341,3,t) array to (1023, t) and then computes the dtw distance between the two embeddings.

        """
        X = np.swapaxes(X.reshape(self.n_embeddings * 3, X.shape[2]), 0, 1)
        Y = np.swapaxes(Y.reshape(self.n_embeddings * 3, Y.shape[2]), 0, 1)
        distance, path = fastdtw(X, Y, dist=euclidean)
        return distance

    def _generateBallTree(self, leaf_size=40):
        """
        This method generates a ball tree for the embedding samples. The ball tree is used to perform knn classification. It uses the custom dtw function as a distance metric.
        """
        dtw_distance_metric = DistanceMetric.get_metric(
            "pyfunc", func=self.dtw_distances
        )
        embedding_samples = [sample.embedding for sample in self._embedding_samples]
        ball_tree = BallTree(
            embedding_samples,
            metric="pyfunc",
            leaf_size=leaf_size,
            metric_params={"n_embeddings": self.n_embeddings},
        )
        return ball_tree

    def knn_dtw_classify(self, embedding_seq, n_neighbors=10):
        """
        This methodes takes a sequence of embeddings and classify the sequence into a class as defined by the class names. It classifies the sequence using a ball tree implementation of knn and uses dtw as a distance metric.
        """


def main():
    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../normalized_embeddings/")
    print("Initializing Action Classifier")
    action_classifier = ActionClassification(embedding_path)
    print("Action Classifier Initialized")
    print(action_classifier._embedding_samples[0].embedding.shape)
    # Reshaping the (341, 3, t) to (1023, t)
    # embedding = embedding.reshape(1023, embedding.shape[2])
    print(
        action_classifier.dtw_distances(
            action_classifier._embedding_samples[0].embedding,
            action_classifier._embedding_samples[1].embedding,
        )
    )
    X_train, X_test, y_train, y_test = train_test_split(
        [sample for sample in action_classifier._embedding_samples],
        [sample.class_name for sample in action_classifier._embedding_samples],
        test_size=0.2,
        random_state=42,
    )
    print("Training")
    print(len(X_test))
    y_pred = []
    start_time = time.process_time()
    for test_sample in X_test:
        distances = []
        for train_sample in X_train:
            distances.append(
                action_classifier.dtw_distances(
                    test_sample.embedding, train_sample.embedding
                )
            )
        y_pred.append(y_train[np.argmin(distances)])
    print("Testing")
    end_time = time.process_time()
    print(f"Time taken: {end_time - start_time}")
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
