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
        print("Embedding samples loaded")
        # self._ball_tree = self._generateBallTree(leaf_size=leaf_size)
        # print("Ball Tree Generated")
        # A custom dtw function using the fast dtw implementation and can be used with sklearn's ball tree

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
    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../embeddings_utd_mhad")
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
