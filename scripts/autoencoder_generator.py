# Create an autoencoder model and save it to a file
"""
It loads the embedding samples, and trains on them frame by frame. It then saves the model to a file.
"""
import scipy.io as sio
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import sys
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../utils/"))
from custom_classes import PoseSample, load_embedding_samples


def per_frame_data(embedding_samples):
    """
    Takes in a list of embedding samples and returns the data in a format that can be used to train the autoencoder
    """
    per_frame_data = []
    for sample in embedding_samples:
        sample_flat = np.swapaxes(
            sample.embedding.reshape(341 * 3, sample.embedding.shape[2]), 0, 1
        )
        print(sample_flat.shape)
        for frame in sample_flat:
            per_frame_data.append(frame)
    print(len(per_frame_data))
    return np.array(per_frame_data)


def create_model(input_shape=(1023,)):
    input_layer = keras.Input(shape=input_shape)

    encoding_layer1 = layers.Dense(512, activation="relu")(input_layer)
    encoding_layer2 = layers.Dense(256, activation="relu")(encoding_layer1)

    bottleneck = layers.Dense(128, activation="relu")(encoding_layer2)

    decoding_layer1 = layers.Dense(256, activation="relu")(bottleneck)
    decoding_layer2 = layers.Dense(512, activation="relu")(decoding_layer1)

    output_layer = layers.Dense(1023, activation="relu")(decoding_layer2)

    model = keras.Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


# def train_model(model, X_train, X_test, y_train, y_test, batch_size=32, epochs=100):
#     checkpoint = ModelCheckpoint(
#         "autoencoder.h5",
#         monitor="val_loss",
#         verbose=1,
#         save_best_only=True,
#         mode="auto",
#         period=1,
#     )
#     callbacks_list = [checkpoint]
#     history = model.fit(
#         X_train,
#         X_train,
#         batch_size=batch_size,
#         epochs=epochs,
#         validation_data=(X_test, X_test),
#         callbacks=callbacks_list,
#     )
#     return history


def plot_loss(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main():
    # Get the path to the embeddings
    currentdir = os.path.dirname(os.path.realpath(__file__))
    embedding_path = os.path.join(currentdir, "../embeddings_utd_mhad")
    print("Loading Data")
    embedding_samples, class_names = load_embedding_samples(embedding_path)
    training_data = per_frame_data(embedding_samples)
    training_data, val_data = train_test_split(
        training_data, test_size=0.2, random_state=42
    )
    print("Data Loaded")
    print("Creating Model")
    model = create_model()
    print("Model Created")
    log_dir = "logs"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_checkpoint_callback = ModelCheckpoint(
        "autoencoder_best_model.h5", save_best_only=True, monitor="val_loss"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        training_data,
        training_data,
        epochs=200,
        batch_size=32,
        shuffle=True,
        validation_data=(val_data, val_data),
        callbacks=[
            tensorboard_callback,
            model_checkpoint_callback,
            early_stopping_callback,
        ],
    )
    # Reconstruct the validation data
    reconstructed_val_data = model.predict(val_data)

    # Compute the mean squared error on the validation data
    mse = np.mean(np.square(val_data - reconstructed_val_data))

    print(f"Mean Squared Error on validation data: {mse}")


if __name__ == "__main__":
    main()
