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
        self, embedding_dir, file_extension="mat", n_clusters=27, leaf_size=40
    ):
        self._embedding_samples, self._class_names = load_embedding_samples(
            embedding_dir=embedding_dir, file_extension=file_extension
        )
        # Preprocess embeddings to have uniform length
        self.embeddings, self.max_length = self.preprocess_embeddings(
            [sample.embedding for sample in self._embedding_samples]
        )
        self.n_clusters = n_clusters
        self.leaf_size = leaf_size
        self.centroids, self.labels = self.compute_dba_centroids(self.embeddings)
        self.ball_tree = self.build_ball_tree()

    def preprocess_embeddings(self, embeddings):
        # Determine maximum length
        max_length = max(embedding.shape[0] for embedding in embeddings)
        # Pad sequences
        padded_embeddings = np.array(
            [self.pad_sequence(embedding, max_length) for embedding in embeddings]
        )
        return padded_embeddings, max_length

    def pad_sequence(self, sequence, max_length):
        # Assuming the sequence shape is (t, 1, 16) and padding is needed across 't'
        padded_sequence = np.zeros((max_length, 1, 16))
        sequence_length = sequence.shape[0]
        padded_sequence[:sequence_length, :, :] = sequence
        return np.squeeze(padded_sequence, axis=1)

    def compute_dba_centroids(self, embeddings):
        # Flatten the embeddings for k-means
        # Reshape back to (n_samples, max_length, 16) for TimeSeriesKMeans
        model = TimeSeriesKMeans(
            n_clusters=self.n_clusters, metric="dtw", max_iter=10, verbose=True
        )
        labels = model.fit_predict(embeddings)
        centroids = model.cluster_centers_
        return centroids, labels

    def build_ball_tree(self):
        # Flatten centroids for the BallTree
        centroids_flat = self.centroids.reshape((self.centroids.shape[0], -1))
        ball_tree = BallTree(centroids_flat, leaf_size=self.leaf_size)
        return ball_tree

    def query_ball_tree(self, sample_embedding):
        sample_flat = sample_embedding.reshape(1, -1)
        dist, ind = self.ball_tree.query(sample_flat, k=1)
        return dist, ind

    def dtw_distance_to_centroid(self, sample_embedding, centroid_index):
        centroid = self.centroids[centroid_index].reshape(sample_embedding.shape)
        distance, _ = dtw(sample_embedding, centroid, distance_only=True)
        return distance.distance


def main():
    embedding_dir = os.path.join(currentdir, "../encoded16_embeddings/")
    print("Initializing Action Classifier")
    classifier = ActionClassificationWithDBA(embedding_dir)
    print("Action Classifier Initialized")

    # Splitting samples for demonstration; in a real scenario, you'd prepare your dataset accordingly.
    X_train, X_test, y_train, y_test = train_test_split(
        classifier._embedding_samples,
        [sample.class_name for sample in classifier._embedding_samples],
        test_size=0.2,
        random_state=42,
    )
    start_time = time.process_time()
    count = 0
    print("Evaluating")
    y_pred = []
    for test_sample in X_test:
        _, nearest_centroid_idx = classifier.query_ball_tree(
            classifier.pad_sequence(test_sample.embedding, classifier.max_length)
        )
        predicted_class = classifier._class_names[nearest_centroid_idx[0][0]]
        y_pred.append(predicted_class)
        print(count)
        count += 1

    print(f"Time taken: {time.process_time() - start_time}")

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
