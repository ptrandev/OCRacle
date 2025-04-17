#
# Model Training Module
#
# This module is responsible for training the model by using a test-train split
# of the dataset and saving the trained model to disk. The ADAM optimizer is used
# to minimize the categorical cross-entropy loss function.
#

import os
import sys
import keras

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data import extractTrainingSamples
from src.model import MODEL_PATH

EPOCHS = 5
BATCH_SIZE = 32

WIDTH = 28
HEIGHT = 28


def train():
    """
    Train the model using the EMNIST dataset.
    The model is a simple feedforward neural network with one hidden layer.
    The model is trained using the ADAM optimizer and the categorical cross-entropy
    loss function.

    The model is saved to disk in the current directory with the name "model.keras".
    """

    # Load EMNIST data
    images, labels = extractTrainingSamples()

    # Define the model
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(WIDTH, HEIGHT)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.1),  # prevent overfitting
            keras.layers.Dense(26, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    print("Training model...")
    model.fit(images, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("Saving model...")
    model.save(MODEL_PATH)


if __name__ == "__main__":  # pragma: no cover
    train()
