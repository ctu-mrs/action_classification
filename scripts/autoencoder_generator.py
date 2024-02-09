import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from custom_classes import PoseSample, load_embedding_samples


def build_autoencoder(input_shape=(1023,)):
    # Encoder
    inputs = Input(shape=input_shape)
    encoded = Dense(512, activation="relu")(inputs)
    encoded = Dense(256, activation="relu")(encoded)
    encoded = Dense(128, activation="relu")(encoded)  # Bottleneck layer

    # Decoder
    decoded = Dense(256, activation="relu")(encoded)
    decoded = Dense(512, activation="relu")(decoded)
    decoded = Dense(input_shape[0], activation="sigmoid")(decoded)

    # Autoencoder
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(), loss="mse")

    return autoencoder


def main():

    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../normalized_embeddings/")
    embedding_samples, _ = load_embedding_samples(
        embedding_dir=embedding_path, file_extension="mat"
    )
    print("Loaded embeddings!")
    print(f"Number of samples: {len(embedding_samples)}")
    flattened_data = []
    for sample in embedding_samples:
        # Convert each embedding to a 1D array
        for i in range(sample.embedding.shape[2]):
            per_frame = sample.embedding[:, :, i]
            flattened_data.append(per_frame.flatten())
    print("Flattened data!")
    print(f"Number of flattened samples: {len(flattened_data)}")
    print(f"Shape of each sample: {flattened_data[0].shape}")
    # Convert to numpy array
    flattened_data = np.array(flattened_data)
    X_train, X_val = train_test_split(flattened_data, test_size=0.2, random_state=42)

    autoencoder = build_autoencoder(input_shape=(1023,))
    print("Built autoencoder model!")
    autoencoder.fit(
        X_train,
        X_train,
        epochs=100,
        batch_size=256,
        shuffle=True,
        validation_data=(X_val, X_val),
    )
    print("Trained autoencoder model!")
    # Check the accuracy of the model
    val_loss = autoencoder.evaluate(X_val, X_val)
    print(f"Validation Loss: {val_loss}")

    # Save the model
    autoencoder.save(os.path.join(currentdir, "autoencoder.keras"))
    print("Autoencoder model saved!")


if __name__ == "__main__":
    main()
