#
# Accuracy Metrics Module
#
# This module is responsible for evaluating the accuracy of the trained model
# using the test dataset.
#

import os
import tensorflow as tf
from emnist import extract_test_samples
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.output import LABELS


# get the accuracy, loss, and confusion matrix
def evaluate(model_path):
    # Load the test dataset
    test_images, test_labels = extract_test_samples("letters")

    # Normalize the images to a pixel range of 0 to 1
    test_images = test_images / 255.0

    # scale labels to 0-25
    test_labels = test_labels - 1

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model, get the cross-entropy loss and accuracy
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)

    # get the confusion matrix
    predictions = model.predict(test_images)
    confusion_matrix = tf.math.confusion_matrix(
        test_labels, tf.argmax(predictions, axis=1)
    )

    # calculate the accuracy of each class, return as a dictionary with class labels
    class_accuracy = {}
    for i in range(len(LABELS)):
        # calculate the accuracy for each class
        class_accuracy[LABELS[i]] = (confusion_matrix[i][i] / tf.reduce_sum(
            confusion_matrix[i]
        )).numpy().item()

    # order class_accuracy by accuracy
    class_accuracy = dict(
        sorted(class_accuracy.items(), key=lambda item: item[1], reverse=True)
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "class_accuracy": class_accuracy,
    }
