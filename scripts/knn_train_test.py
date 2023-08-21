import scipy.io as sio
import numpy as np
import os
from fastdtw import fastdtw
from sklearn.neighbors import BallTree
from feature_vector_generator import FeatureVectorEmbedder
from scipy.spatial.distance import euclidean


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
        self._ball_tree = self._generateBallTree(leaf_size=leaf_size)
        print("Ball Tree Generated")

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
            metric=self._dtw_distances,
        )
        return tree

    def knn_dtw_classify(self, embedding_seq, n_neighbors=10):
        """
        This methodes takes a sequence of embeddings and classify the sequence into a class as defined by the class names. It classifies the sequence using a ball tree implementation of knn and uses dtw as a distance metric.
        """

    def _dtw_distances(self, seq1, seq2):
        seq1_flat = [matrix.flatten() for matrix in seq1]
        seq2_flat = [matrix.flatten() for matrix in seq2]
        return fastdtw(seq1_flat, seq2_flat, dist=euclidean)

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


if __name__ == "__main__":
    main()
