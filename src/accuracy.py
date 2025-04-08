#
# Accuracy Metrics Module
#
# This module is responsible for evaluating the accuracy of the trained model
# using the test dataset.
#

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.output import LABELS
from src.data import extractTestSamples
from src.model import MODEL


# get the accuracy, loss, and confusion matrix
def evaluate():
    """
    Evaluate the model using the test dataset.

    Args:
        model_path (str): The path to the trained model.

    Returns:
        loss (float): The cross-entropy loss of the model on the test dataset.
        accuracy (float): The accuracy of the model on the test dataset.
        confusionMatrix (tf.Tensor): The confusion matrix of the model on the test dataset.
        classAccuracy (Dict[str, float]): The accuracy of each class in the test dataset.
    """

    # Load the test dataset
    test_images, test_labels = extractTestSamples()

    # Evaluate the model, get the cross-entropy loss and accuracy
    loss, accuracy = MODEL.evaluate(test_images, test_labels, verbose="0")

    # get the confusion matrix
    predictions = MODEL.predict(test_images)
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


if __name__ == "__main__":
    # Evaluate the model
    evaluation = evaluate() #

    # Print the evaluation results
    print("Loss:", evaluation["loss"])
    print("Accuracy:", evaluation["accuracy"])
    print("Confusion Matrix:\n", evaluation["confusionMatrix"])
    print("Class Accuracy:\n", evaluation["classAccuracy"])