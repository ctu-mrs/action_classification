import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import sys
import time
from dtw import dtw

# Assuming custom_classes.py exists in the "../utils/" directory and provides
# PoseSample and load_embedding_samples functions
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from custom_classes import PoseSample, load_embedding_samples


class ActionClassificationWithDBA(object):
    def __init__(
        self, embedding_dir, file_extension="mat", n_clusters=10, leaf_size=40
    ):
        self._embedding_samples, self._class_names = load_embedding_samples(
            embedding_dir=embedding_dir, file_extension=file_extension
        )
        # A dictionary to map embeddings to class names
        self._index_to_class = {
            emb_idx: sample.class_name
            for emb_idx, sample in enumerate(self._embedding_samples)
        }

        self.embeddings, self.labels, self.max_length = None, None, None
        self.leaf_size = leaf_size
        self.ball_tree, self.centroids = None, None

    def preprocess_and_cluster(self, embeddings, n_clusters):
        max_length = max(embedding.shape[0] for embedding in embeddings)
        padded_embeddings = np.array(
            [self.pad_sequence(embedding, max_length) for embedding in embeddings]
        )
        model = TimeSeriesKMeans(
            n_clusters=n_clusters, metric="dtw", max_iter=100, verbose=False
        )
        labels = model.fit_predict(padded_embeddings)
        self.centroids = model.cluster_centers_
        return padded_embeddings, labels, max_length

    def pad_sequence(self, sequence, max_length):
        # Assuming the sequence shape is (t, 1, 16) and padding is needed across 't'
        padded_sequence = np.zeros((max_length, 1, 16))
        sequence_length = sequence.shape[0]
        padded_sequence[:sequence_length, :, :] = sequence
        return np.squeeze(padded_sequence, axis=1)

    def build_ball_tree(self):
        centroids_flat = self.centroids.reshape((self.centroids.shape[0], -1))
        ball_tree = BallTree(centroids_flat, leaf_size=self.leaf_size)
        return ball_tree, centroids_flat

    def query_ball_tree(self, sample_embedding, n_candidates=1):
        sample_flat = sample_embedding.reshape(1, -1)
        distances, indices = self.ball_tree.query(sample_flat, k=n_candidates)
        return distances, indices

    def find_nearest_neighbors_dtw(self, sample_embedding, n_neighbors=5):
        distances, indices = self.query_ball_tree(sample_embedding, n_candidates=1)
        nearest_neighbors = []

        for idx in indices[0]:
            cluster_embeddings_indexes = np.where(self.labels == idx)[0]

            for index_of_candidate in cluster_embeddings_indexes:
                candidate = self.embeddings[index_of_candidate]
                dtw_distance = dtw(
                    sample_embedding, candidate, distance_only=True
                ).distance
                candidate_class_name = self._index_to_class[index_of_candidate]
                nearest_neighbors.append((dtw_distance, candidate_class_name))
        nearest_neighbors.sort(key=lambda x: x[0])
        print("Nearest Neighbors", nearest_neighbors[:n_neighbors])
        return nearest_neighbors[:n_neighbors]


def main():
    embedding_dir = os.path.join(currentdir, "../encoded16_embeddings/")
    print("Initializing Action Classifier")
    classifier = ActionClassificationWithDBA(embedding_dir)
    print("Action Classifier Initialized")

    # Prepare your dataset: load, split, etc.
    X_train, X_test, y_train, y_test = train_test_split(
        [
            sample.embedding for sample in classifier._embedding_samples
        ],  # Use embeddings directly
        [sample.class_name for sample in classifier._embedding_samples],
        test_size=0.2,
        random_state=42,
    )
    print(len(X_train), len(X_test))
    classifier.embeddings, classifier.labels, classifier.max_length = (
        classifier.preprocess_and_cluster(
            embeddings=[sample for sample in X_train],
            n_clusters=5,
        )
    )
    classifier.ball_tree, classifier.centroids = classifier.build_ball_tree()
    start_time = time.process_time()
    print("Evaluating")
    y_pred = []
    for idx_test, test_embedding in enumerate(X_test):
        nearest_neighbors = classifier.find_nearest_neighbors_dtw(
            classifier.pad_sequence(test_embedding, classifier.max_length),
            n_neighbors=5,  # Adjust based on your requirements
        )
        print("The expected class is: ", y_test[idx_test])
        # Predict the class based on nearest neighbors
        # This example simply takes the mode of the nearest neighbor classes; adjust as needed
        nearest_classes = [neighbor[1] for neighbor in nearest_neighbors]
        predicted_class = max(set(nearest_classes), key=nearest_classes.count)
        y_pred.append(predicted_class)

    print(f"Time taken: {time.process_time() - start_time}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
