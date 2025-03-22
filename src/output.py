#
# Model Output Module
#
# This module is responsible for processing the model's output and displaying it
# to the user in a human readable format.
#

from src.model import predict
import numpy as np

LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


def output(imageMatrix: np.ndarray) -> dict[str, np.ndarray]:
    """
    Processes the model's output and displays it to the user in a human readable format.

    Parameters:
    imageMatrix (np.ndarray): The matrix representation of the image

    Returns:
    tuple (str, np.ndarray): The predicted label and the model's output
    """

    prediction = predict(imageMatrix)

    return {
        "predictedLabel": LABELS[np.argmax(prediction) - 1],
        "confidenceMatrix": prediction,
    }
