#
# Model Training Module
#
# This module is responsible for training the model by using a test-train split
# of the dataset and saving the trained model to disk. The ADAM optimizer is used
# to minimize the categorical cross-entropy loss function.
#

import os
import tensorflow as tf
from emnist import extract_training_samples, extract_test_samples

EPOCHS = 5
BATCH_SIZE = 32


def train():
    # Load EMNIST data
    images, labels = extract_training_samples("letters")
    test_images, test_labels = extract_test_samples("letters")

    # Normalize the images to a pixel range of 0 to 1
    images = images / 255.0
    test_images = test_images / 255.0

    # scale labels to 0-25
    labels = labels - 1
    test_labels = test_labels - 1

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

    print("Evaluating model...")
    model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)

    print("Saving model...")
    model.save(os.path.join(os.path.dirname(__file__), "model.keras"))


train()
