import os
import scipy.io as sio
import numpy as np


class PoseSample(object):
    def __init__(self, name, class_name, embedding):
        self.name = name
        self.class_name = class_name
        self.embedding = embedding

    def __repr__(self) -> str:
        return f"PoseSample(name={self.name}, class_name={self.class_name}, embedding={self.embedding})"


class Utilities(object):
    def _load_embedding_samples(self, embedding_dir, file_extension="mat"):
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
        print("Embedding samples loaded using Utilities")
        return embedding_samples, class_names
