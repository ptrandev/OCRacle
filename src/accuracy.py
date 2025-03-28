#
# Accuracy Metrics Module
#
# This module is responsible for evaluating the accuracy of the trained model
# using the test dataset.
#

import os
import sys
import tensorflow as tf
import keras
from typing import cast, Dict, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.output import LABELS
from src.data import extractTestSamples


# get the accuracy, loss, and confusion matrix
def evaluate(model_path: str) -> Dict[str, Union[float, tf.Tensor, Dict[str, float]]]:
    # Load the test dataset
    test_images, test_labels = extractTestSamples()

    # Load the trained model
    model = cast(keras.Model, keras.models.load_model(model_path))

    # Evaluate the model, get the cross-entropy loss and accuracy
    loss, accuracy = model.evaluate(test_images, test_labels, verbose="0")

    # get the confusion matrix
    predictions = model.predict(test_images)
    confusionMatrix = tf.math.confusion_matrix(
        test_labels, tf.argmax(predictions, axis=1)
    )

    # calculate the accuracy of each class, return as a dictionary with class labels
    classAccuracy = {}
    for i in range(len(LABELS)):
        # calculate the accuracy for each class
        classAccuracy[LABELS[i]] = (
            (confusionMatrix[i][i] / tf.reduce_sum(confusionMatrix[i])).numpy().item()
        )

    # order classAccuracy by accuracy
    classAccuracy = dict(
        sorted(classAccuracy.items(), key=lambda item: item[1], reverse=True)
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "confusionMatrix": confusionMatrix,
        "classAccuracy": classAccuracy,
    }
