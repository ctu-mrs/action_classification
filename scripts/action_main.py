#!/usr/bin/env python3
import scipy.io as sio
import os
import numpy as np
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from custom_classes import PoseSample, load_embedding_samples, mean_vector
from sklearn.neighbors import BallTree
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class ActionClassification(object):
    def __init__(self, embedding_samples, class_names):
        self._embedding_samples, self._class_names = embedding_samples, class_names
        self._class_names = np.unique(
            [sample.class_name for sample in self._embedding_samples]
        )
        self._n_embeddings = 341
        self._n_dimensions = 3
        self._sliding_window_size = 30
        self._n_neighbors = 10
        self.mean_embedding_samples = []
        for class_name in self._class_names:
            embedding_samples = [
                sample.embedding
                for sample in self._embedding_samples
                if sample.class_name == class_name
            ]
            mean_embedding = mean_vector(embedding_samples)
            self.mean_embedding_samples.append(
                PoseSample(
                    name=f"{class_name}_mean",
                    class_name=class_name,
                    embedding=mean_embedding,
                )
            )
        self._generateBallTree()

    def _generateBallTree(self, leaf_size=40):
        "Generate a Ball Tree using the mean embeddings"
        self.ball_tree = BallTree(
            [sample.embedding for sample in self.mean_embedding_samples],
            leaf_size=leaf_size,
        )

    def embedding_filter(self, embedding, filter_limit=50):
        "Uses ball tree to find the closest embeddings to the input embedding"
        distances, indices = self.ball_tree.query(embedding, k=filter_limit)
        return [self.mean_embedding_samples[index] for index in indices[0]]

    def dtw_distances(self, X, Y):
        """
        This method takes in two embeddings, flattens the (341,3,t) array to (1023, t) and then computes the dtw distance between the two embeddings.

        """
        X = np.swapaxes(
            X.reshape(self._n_embeddings * self._n_dimensions, X.shape[2]), 0, 1
        )
        Y = np.swapaxes(
            Y.reshape(self._n_embeddings * self._n_dimensions, Y.shape[2]), 0, 1
        )
        distance, path = fastdtw(X, Y, dist=euclidean)
        return distance

    def classify(self, input_embedding):
        # Step 1: Retrieve initial candidate set using BallTree
        candidates = self.embedding_filter(
            input_embedding, filter_limit=self._n_neighbors
        )

        # Step 2: Refine candidates using DTW
        refined_candidates = []
        for candidate in candidates:
            dtw_distance = self.dtw_distances(input_embedding, candidate.embedding)
            if (
                dtw_distance < self.max_dtw_threshold
            ):  # You need to define this threshold
                refined_candidates.append((candidate, dtw_distance))

        # Sort by DTW distance (you can choose max or mean as per your requirement)
        refined_candidates.sort(key=lambda x: x[1])

        # Step 3: Final classification decision
        if not refined_candidates:
            return None  # or some default class

        # Get the class names of the top refined candidates
        top_classes = [
            candidate[0].class_name
            for candidate in refined_candidates[: self._n_neighbors]
        ]

        # Count the frequency of each class
        class_counts = Counter(top_classes)

        # Determine the most frequent class
        most_common_class = class_counts.most_common(1)[0][0]

        return most_common_class


def main():
    # Get the path to the embeddings
    embedding_dir = os.path.join(currentdir, "../data/")
    embedding_samples, class_names = load_embedding_samples(
        embedding_dir=embedding_dir, file_extension="mat"
    )
    # Split into train and test
    train_samples = embedding_samples[: int(0.8 * len(embedding_samples))]
    test_samples = embedding_samples[int(0.8 * len(embedding_samples)) :]

    # Create an instance of the ActionClassification class
    action_classifier = ActionClassification(train_samples, class_names)

    # Test The classifier and print the accuracy, confusion matrix and classification report
    y_true = []
    y_pred = []
    for sample in test_samples:
        predicted_class = action_classifier.classify(sample.embedding)
        y_pred.append(predicted_class)
        y_true.append(sample.class_name)
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # # Test the classifier on the test samples
    # correct = 0
    # for sample in test_samples:
    #     predicted_class = action_classifier.classify(sample.embedding)
    #     if predicted_class == sample.class_name:
    #         correct += 1
    #     else:
    #         print(
    #             f"Predicted class: {predicted_class}, Actual class: {sample.class_name}"
    #         )
    # print(f"Accuracy: {correct/len(test_samples)}")


if __name__ == "__main__":
    main()
