#
# Model Training Module
#
# This module is responsible for training the model by using a test-train split
# of the dataset and saving the trained model to disk. The ADAM optimizer is used
# to minimize the categorical cross-entropy loss function.
#

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data import extractTrainingSamples

EPOCHS = 5
BATCH_SIZE = 32


def train():
    # Load EMNIST data
    images, labels = extractTrainingSamples()

    # Define the model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(26, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    print("Training model...")
    model.fit(images, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("Saving model...")
    model.save(os.path.join(os.path.dirname(__file__), "model.keras"))


train()
