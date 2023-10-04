import scipy.io as sio
import numpy as np
import os
from fastdtw import fastdtw
from feature_vector_generator import FeatureVectorEmbedder
from scipy.spatial.distance import euclidean
from sklearn.metrics import DistanceMetric
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from multiprocessing import Pool
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


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

    def dtw_distances(self, X, Y):
        """
        This method takes in two embeddings, flattens the (341,3,t) array to (1023, t) and then computes the dtw distance between the two embeddings.

        """
        X = np.swapaxes(X.reshape(self.n_embeddings * 3, X.shape[2]), 0, 1)
        Y = np.swapaxes(Y.reshape(self.n_embeddings * 3, Y.shape[2]), 0, 1)
        distance, _ = fastdtw(X, Y, dist=euclidean)
        return distance

    def parallel_dtw(self, X, Y):
        with Pool() as p:
            distances = p.starmap(self.dtw_distances, [(x, y) for x, y in zip(X, Y)])
        return np.array(distances).reshape((len(X), len(Y)))

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
    X_train, X_test, y_train, y_test = train_test_split(
        [sample for sample in action_classifier._embedding_samples],
        [sample.class_name for sample in action_classifier._embedding_samples],
        test_size=0.2,
        random_state=42,
    )
    print("Training")
    print(len(X_test))
    nn = NearestNeighbors(
        metric=action_classifier.parallel_dtw, n_jobs=1
    )  # Note: n_jobs is set to 1

    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)
    y_pred = y_train[indices[:, 0]]  # assuming labels are in y_train

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
