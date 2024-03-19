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
from dtw import *

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from custom_classes import PoseSample


def load_embedding_samples(embedding_dir, file_extension="mat"):
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
                embedding = data["d_skel"]
                # The class is the name of the sub folder that contains the .mat file
                class_name = os.path.basename(root)
                embedding_samples.append(PoseSample(file, class_name, embedding))
    class_names = np.unique([sample.class_name for sample in embedding_samples])
    print("Embedding samples loaded using Utilities")
    return embedding_samples, class_names


class ActionClassification(object):
    def __init__(
        self,
        embedding_dir,
        leaf_size=40,
        file_extension="mat",
        n_embeddings=20,
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
        Calculates DTW distances between two samples. The samples are of (t,1,128) shape. The samples are reshaped to (t,128) and then the fastdtw function is used to calculate the distance.
        """
        # X = np.squeeze(X, axis=1)
        # Y = np.squeeze(Y, axis=1)
        X = np.swapaxes(X.reshape(self.n_embeddings * 3, X.shape[2]), 0, 1)
        Y = np.swapaxes(Y.reshape(self.n_embeddings * 3, Y.shape[2]), 0, 1)
        # distance, path = fastdtw(X, Y, dist=euclidean, radius=1)
        alignment = dtw(X, Y, keep_internals=True)
        distance = alignment.distance
        return distance


def main():
    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../utd_mhad_dataset/")
    print("Initializing Action Classifier")
    action_classifier = ActionClassification(embedding_path)
    print("Action Classifier Initialized")
    print(action_classifier._embedding_samples[0].embedding.shape)
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
    count = 0
    for test_sample in X_test:
        distances = []
        for train_sample in X_train:
            distances.append(
                action_classifier.dtw_distances(
                    test_sample.embedding, train_sample.embedding
                )
            )
        y_pred.append(y_train[np.argmin(distances)])
        count += 1
        print(count)
    end_time = time.process_time()
    print(f"Time taken: {end_time - start_time}")
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
