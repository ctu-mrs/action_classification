"""
Load the embeddings which are in the form of  a matrix of shape n_features x n_dims x n_frames). Calculate the average of the embeddings across the frames to get a single vector of shape n_features x n_dims. This is the vector that will be used to train the autoencoder. It then applies z-score normalization to the embeddings and export normalized embeddings to a file.
"""

import scipy.io as sio
import os
import numpy as np

import matplotlib.pyplot as plt


class PoseSample(object):
    def __init__(self, name, class_name, embedding):
        self.name = name
        self.class_name = class_name
        self.embedding = embedding

    def __repr__(self) -> str:
        return f"PoseSample(name={self.name}, class_name={self.class_name}, embedding={self.embedding})"


def _load_embedding_samples(embedding_dir, file_extension):
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
    embedding_samples, class_names = _load_embedding_samples(embedding_path, "mat")
    print("Embeddings Loaded")

    # Calculate the average and sd of each feature across the frames. Each embedding sample might contain different number of frames. So we need to calculate the average and sd of each feature across the frames for each embedding sample.

    # Calculate the average and sd of each feature across the frames for each embedding sample

    average_embeddings = []
    sd_embeddings = []
    for sample in embedding_samples:
        average_embedding = np.mean(sample.embedding, axis=2)
        sd_embedding = np.std(sample.embedding, axis=2)
        sd_embeddings.append(sd_embedding)
        average_embeddings.append(average_embedding)
    average_embeddings = np.array(average_embeddings)
    sd_embeddings = np.array(sd_embeddings)
    print(average_embeddings.shape)
    print(sd_embeddings.shape)

    # Z-score normalization
    normalized_embeddings = []
    for idx, sample in enumerate(embedding_samples):
        normalized_embedding = np.zeros(sample.embedding.shape)
        for i in range(sample.embedding.shape[2]):
            normalized_embedding[:, :, i] = (
                sample.embedding[:, :, i] - average_embeddings[idx]
            ) / sd_embeddings[idx]
        normalized_embeddings.append(normalized_embedding)

    # Verify that the normalization worked
    verify_normalization(normalized_embeddings)
    # Write the normalized embeddings to a file in the same format as the original embeddings and as the same folder structure
    normalized_embedding_path = os.path.join(currentdir, "../normalized_embeddings")
    for sample in embedding_samples:
        # Get the parent directory of the mat file
        parent_dir = os.path.dirname(sample.name)

        # Get the name of the mat file without the extension
        file_name = os.path.basename(sample.name).split(".")[0]

        # Create the output directory if it doesn't exist
        if not os.path.exists(normalized_embedding_path):
            os.mkdir(normalized_embedding_path)

        # Create the output parent directory
        new_parent_dir = os.path.join(
            normalized_embedding_path, parent_dir.replace("embeddings_utd_mhad/", "", 1)
        )
        if not os.path.exists(new_parent_dir):
            os.makedirs(new_parent_dir)

        # Save the mat file to the output directory
        new_mat_file_path = os.path.join(new_parent_dir, file_name + ".mat")
        sio.savemat(new_mat_file_path, {"embedding": normalized_embedding})


if __name__ == "__main__":
    main()
