import numpy as np
import os
import sys
import time
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tslearn.barycenters import dtw_barycenter_averaging
from dtw import dtw

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from custom_classes import PoseSample, load_embedding_samples


class ActionClassificationWithDBA(object):
    def __init__(self, embedding_dir, file_extension="mat"):
        self._embedding_samples, _ = load_embedding_samples(
            embedding_dir=embedding_dir, file_extension=file_extension
        )
        self.labels = np.array(
            [sample.class_name for sample in self._embedding_samples]
        )
        self.embeddings = np.array(
            [sample.embedding for sample in self._embedding_samples]
        )
        self.classes = np.unique(self.labels)
        self.class_to_index = {
            label: np.where(self.labels == label)[0] for label in self.classes
        }
        self.centroids = None
        self.ball_tree = None
        self.leaf_size = 40

    def pad_sequence(self, sequence, max_length):
        # Assuming the sequence shape is (t, 1, 16) and padding is needed across 't'
        padded_sequence = np.zeros((max_length, 1, 16))
        sequence_length = sequence.shape[0]
        padded_sequence[:sequence_length, :, :] = sequence
        return np.squeeze(padded_sequence, axis=1)

    def compute_dba_centroids(self):
        self.centroids = np.array(
            [
                dtw_barycenter_averaging(self.embeddings[self.class_to_index[label]])
                for label in self.classes
            ]
        )

    def build_ball_tree(self):
        self.compute_dba_centroids()
        # Flatten centroids for the ball tree
        centroids_flat = np.array([c.flatten() for c in self.centroids])
        self.ball_tree = BallTree(centroids_flat, leaf_size=self.leaf_size)

    def find_nearest_clusters(self, sample_embedding, n_clusters=5):
        sample_flat = sample_embedding.flatten().reshape(1, -1)
        distances, indices = self.ball_tree.query(sample_flat, k=n_clusters)
        return indices[0]

    def classify_sample(self, sample_embedding, n_neighbors=5):
        nearest_cluster_idxs = self.find_nearest_clusters(sample_embedding)
        nearest_neighbors = []

        for idx in nearest_cluster_idxs:
            class_name = self.classes[idx]
            class_indices = self.class_to_index[class_name]
            for index in class_indices:
                candidate = self.embeddings[index]
                dtw_distance = dtw(
                    sample_embedding, candidate, distance_only=True
                ).distance
                nearest_neighbors.append((dtw_distance, class_name))

        nearest_neighbors.sort(key=lambda x: x[0])
        # Return the class of the closest embedding
        return nearest_neighbors[:n_neighbors]


def main():
    embedding_dir = os.path.join(currentdir, "../encoded16_embeddings/")
    classifier = ActionClassificationWithDBA(embedding_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        classifier.embeddings, classifier.labels, test_size=0.2, random_state=42
    )

    classifier.build_ball_tree()

    y_pred = []
    for embedding in X_test:
        neighbors = classifier.classify_sample(embedding)
        # Predict based on the most common class among nearest neighbors
        predicted_classes = [neighbor[1] for neighbor in neighbors]
        predicted_class = max(set(predicted_classes), key=predicted_classes.count)
        y_pred.append(predicted_class)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
