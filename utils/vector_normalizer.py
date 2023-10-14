"""
Load the embeddings which are in the form of  a matrix of shape n_features x n_dims x n_frames). Calculate the average of the embeddings across the frames to get a single vector of shape n_features x n_dims. This is the vector that will be used to train the autoencoder. It then applies z-score normalization to the embeddings and export normalized embeddings to a file.
"""

from typing import Any
import scipy.io as sio
import os
import numpy as np
from custom_classes import PoseSample, Utilities
import matplotlib.pyplot as plt


class VectorNormalizer(object):
    def __init__(
        self,
        embedding_dir,
        file_extension="mat",
        n_embeddings=341,
        n_dimensions=3,
        sliding_window_size=30,
        n_neighbors=10,
    ):
        self._embedding_dir = embedding_dir
        self.n_embeddings = n_embeddings
        self.n_dimensions = n_dimensions
        self.sliding_window_size = sliding_window_size
        self.n_neighbors = n_neighbors
        utils = Utilities()
        self._embedding_samples, self._class_names = utils._load_embedding_samples(
            embedding_dir, file_extension
        )

    def _calculate_parameters(self, embedding_samples):
        """
        Calculates the average, sd, min and max of all the features across the frames for each embedding sample.
        """
        concatenated_embeddings = []
        for sample in embedding_samples:
            for frame_idx in range(sample.embedding.shape[2]):
                features_per_frame = sample.embedding[:, :, frame_idx]
                concatenated_embeddings.append(features_per_frame)

        concatenated_embeddings = np.array(concatenated_embeddings)
        self.average_embeddings = np.mean(concatenated_embeddings, axis=0)
        self.sd_embeddings = np.std(concatenated_embeddings, axis=0)
        self.min_embeddings = np.min(concatenated_embeddings, axis=0)
        self.max_embeddings = np.max(concatenated_embeddings, axis=0)

        # Plot the average, sd, min and max of all the features across the frames for each embedding sample.
        # self._plot_parameters(
        # average_embeddings, sd_embeddings, min_embeddings, max_embeddings
        # )

    # def _plot_parameters(
    #     self, average_embeddings, sd_embeddings, min_embeddings, max_embeddings
    # ):
    #     """
    #     Plots the average, sd, min and max of all the features across the frames for each embedding sample.
    #     """
    #     # Plot the average of all the features across the frames for each embedding sample.
    #     plt.figure()
    #     plt.title(
    #         "Average of all the features across the frames for each embedding sample"
    #     )
    #     plt.plot(average_embeddings)
    #     plt.savefig("average_embeddings.png")

    #     # Plot the sd of all the features across the frames for each embedding sample.
    #     plt.figure()
    #     plt.title("SD of all the features across the frames for each embedding sample")
    #     plt.plot(sd_embeddings)
    #     plt.savefig("sd_embeddings.png")

    #     # Plot the min of all the features across the frames for each embedding sample.
    #     plt.figure()
    #     plt.title("Min of all the features across the frames for each embedding sample")
    #     plt.plot(min_embeddings)
    #     plt.savefig("min_embeddings.png")

    #     # Plot the max of all the features across the frames for each embedding sample.
    #     plt.figure()
    #     plt.title("Max of all the features across the frames for each embedding sample")
    #     plt.plot(max_embeddings)
    #     plt.savefig("max_embeddings.png")

    # Takes the embedding and rertuns the normalized embedding
    def _normalize_embedding(self, sample, normalization_method="minmax"):
        if normalization_method == "minmax":
            normalized_embedding = np.zeros(sample.shape)
            print(sample.shape)
            for i in range(sample.shape[2]):
                normalized_embedding[:, :, i] = (
                    2
                    * (
                        (sample[:, :, i] - self.min_embeddings)
                        / (self.max_embeddings - self.min_embeddings)
                    )
                    - 1
                )
        elif normalization_method == "zscore":
            normalized_embedding = np.zeros(sample.shape)
            for i in range(sample.shape[2]):
                normalized_embedding[:, :, i] = (
                    sample[:, :, i] - self.average_embeddings
                ) / self.sd_embeddings
        return normalized_embedding

    def _save_normalized_embeddings(self, normalized_embedding_path):
        for root, directories, files in os.walk(self._embedding_dir):
            for file in files:
                if file.endswith(".mat"):
                    mat_file_path = os.path.join(root, file)
                    data = sio.loadmat(mat_file_path)
                    embedding = data["embedding"]
                    normalized_embeddings = self._normalize_embedding(embedding)

                    # Get the parent directory of the mat file
                    parent_dir = os.path.dirname(self._embedding_dir)
                    file_name = os.path.basename(file).split(".")[0]

                    # Create the output directory if it doesn't exist
                    if not os.path.exists(normalized_embedding_path):
                        os.mkdir(normalized_embedding_path)

                    # Create the output parent directory
                    new_parent_dir = os.path.join(
                        normalized_embedding_path,
                        parent_dir.replace("embeddings_utd_mhad/", "", 1),
                    )
                    if not os.path.exists(new_parent_dir):
                        os.makedirs(new_parent_dir)

                    # Save the mat file to the output directory
                    new_mat_file_path = os.path.join(new_parent_dir, file_name + ".mat")
                    sio.savemat(new_mat_file_path, {"embedding": normalized_embeddings})
                    print(f"Saved normalized embedding to {new_mat_file_path}")


def verify_normalization(normalized_embeddings):
    # Concatenate all normalized embeddings into one large array for verification
    all_embeddings = np.concatenate(
        [sample for sample in normalized_embeddings], axis=2
    )
    # Compute mean and standard deviation across frames for each feature
    mean_values = np.mean(all_embeddings, axis=2)
    std_values = np.std(all_embeddings, axis=2)

    # Output the mean and standard deviation values
    print(f"Mean values:\n{mean_values}")
    print(f"Standard Deviation values:\n{std_values}")

    # Optionally, you could assert that the mean is close to 0 and standard deviation is close to 1
    np.testing.assert_allclose(
        mean_values, 0, atol=1e-7, err_msg="Mean is not close to 0"
    )
    np.testing.assert_allclose(
        std_values, 1, atol=1e-7, err_msg="Standard deviation is not close to 1"
    )


def main():
    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../embeddings_utd_mhad")
    print("Loading Embeddings")
    vector_normalizer = VectorNormalizer(embedding_path)
    embedding_samples = vector_normalizer._embedding_samples
    vector_normalizer._calculate_parameters(embedding_samples)
    normalized_embedding_path = os.path.join(currentdir, "../normalized_embeddings")
    vector_normalizer._save_normalized_embeddings(normalized_embedding_path)
    # Calculate the average and sd of each feature across the frames. Each embedding sample might contain different number of frames. So we need to calculate the average and sd of each feature across the frames for each embedding sample.

    # Calculate the average and sd of each feature across the frames for each embedding sample

    # average_embeddings = []
    # sd_embeddings = []
    # for sample in embedding_samples:
    #     average_embedding = np.mean(sample.embedding, axis=2)
    #     sd_embedding = np.std(sample.embedding, axis=2)
    #     sd_embeddings.append(sd_embedding)
    #     average_embeddings.append(average_embedding)
    # average_embeddings = np.array(average_embeddings)
    # sd_embeddings = np.array(sd_embeddings)
    # print(average_embeddings.shape)
    # print(sd_embeddings.shape)

    # # Z-score normalization
    # normalized_embeddings = []
    # for idx, sample in enumerate(embedding_samples):
    #     normalized_embedding = np.zeros(sample.embedding.shape)
    #     for i in range(sample.embedding.shape[2]):
    #         normalized_embedding[:, :, i] = (
    #             sample.embedding[:, :, i] - average_embeddings[idx]
    #         ) / sd_embeddings[idx]
    #     normalized_embeddings.append(normalized_embedding)

    # # Verify that the normalization worked
    # verify_normalization(normalized_embeddings)
    # # Write the normalized embeddings to a file in the same format as the original embeddings and as the same folder structure
    # normalized_embedding_path = os.path.join(currentdir, "../normalized_embeddings")
    # for sample in embedding_samples:
    #     # Get the parent directory of the mat file
    #     parent_dir = os.path.dirname(sample.name)

    #     # Get the name of the mat file without the extension
    #     file_name = os.path.basename(sample.name).split(".")[0]

    #     # Create the output directory if it doesn't exist
    #     if not os.path.exists(normalized_embedding_path):
    #         os.mkdir(normalized_embedding_path)

    #     # Create the output parent directory
    #     new_parent_dir = os.path.join(
    #         normalized_embedding_path, parent_dir.replace("embeddings_utd_mhad/", "", 1)
    #     )
    #     if not os.path.exists(new_parent_dir):
    #         os.makedirs(new_parent_dir)

    #     # Save the mat file to the output directory
    #     new_mat_file_path = os.path.join(new_parent_dir, file_name + ".mat")
    #     sio.savemat(new_mat_file_path, {"embedding": normalized_embedding})


if __name__ == "__main__":
    main()
