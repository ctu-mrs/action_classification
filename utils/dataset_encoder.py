import scipy.io as sio
import os
import numpy as np
from custom_classes import PoseSample, load_embedding_samples
from tensorflow.keras.models import load_model, Model
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../scripts/"))


def main():

    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../embeddings_utd_mhad")
    encoded_path = os.path.join(currentdir, "../encoded16_embeddings")
    autoencoder_path = os.path.join(currentdir, "../scripts/")
    autoencoder = load_model(autoencoder_path + "autoencoder_16.keras")
    print("Loaded autoencoder model!")
    encoder_layer_index = 6
    encoder = Model(
        inputs=autoencoder.input, outputs=autoencoder.layers[encoder_layer_index].output
    )

    for root, dirs, files in os.walk(embedding_path):
        for file in files:
            if file.endswith(".mat"):
                mat_file_path = os.path.join(root, file)
                data = sio.loadmat(mat_file_path)
                embedding = data["embedding"]
                encoded_embedding = []
                for i in range(embedding.shape[2]):
                    per_frame = embedding[:, :, i]
                    flattened_data = per_frame.flatten()
                    flattened_data = np.expand_dims(flattened_data, axis=0)
                    encoded_embedding.append(encoder.predict(flattened_data))
                encoded_embedding = np.array(encoded_embedding)
                print(f"Original embedding shape: {embedding.shape}")
                print(f"Encoded embedding shape: {encoded_embedding.shape}")

                relative_file_path = os.path.relpath(root, embedding_path)

                new_dir_path = os.path.join(encoded_path, relative_file_path)
                os.makedirs(new_dir_path, exist_ok=True)

                new_mat_file_path = os.path.join(new_dir_path, file)

                sio.savemat(new_mat_file_path, {"embedding": encoded_embedding})
                print(f"Saved encoded embedding to {new_mat_file_path}")


if __name__ == "__main__":
    main()
