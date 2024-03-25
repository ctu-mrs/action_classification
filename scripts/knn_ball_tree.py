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
        self,
        embedding_dir,
        file_extension="mat",
        n_clusters=10,
        leaf_size=40,
    ):
        self._embedding_samples, self._class_names = load_embedding_samples(
            embedding_dir=embedding_dir, file_extension=file_extension
        )
        self.n_clusters = n_clusters
        self.leaf_size = leaf_size
        self.centroids, self.labels = self.compute_dba_centroids()
        self.ball_tree = self.build_ball_tree()

    def pad_sequences(
        sequences,
        maxlen=None,
        dtype="float32",
        padding="post",
        truncating="post",
        value=0.0,
    ):
        """Pads sequences to the same length."""
        lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
        num_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # Initialize the array to be returned
        x = np.full((num_samples, maxlen, sequences[0].shape[-1]), value, dtype=dtype)
        for i, s in enumerate(sequences):
            if len(s) == 0:
                continue  # Skip this sample because it's empty
            if truncating == "pre":
                trunc = s[-maxlen:]
            else:
                trunc = s[:maxlen]

            # Check if we need to truncate at all
            trunc = np.asarray(trunc, dtype=dtype)
            if padding == "post":
                x[i, : len(trunc)] = trunc
            else:
                x[i, -len(trunc) :] = trunc
        return x

    def compute_dba_centroids(self):
        embeddings = [sample.embedding for sample in self._embedding_samples]
        embeddings_padded = [
            self.pad_sequences(sample.embedding) for sample in self._embedding_samples
        ]
        # Now embeddings_padded is a uniformly shaped array suitable for TimeSeriesKMeans
        model = TimeSeriesKMeans(
            n_clusters=self.n_clusters, metric="dtw", verbose=False
        )
        labels = model.fit_predict(embeddings_padded)
        centroids = model.cluster_centers_
        return centroids, labels

    def build_ball_tree(self):
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
            test_sample.embedding.reshape(-1)
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
