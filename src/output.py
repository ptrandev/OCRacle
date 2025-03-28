#
# Model Output Module
#
# This module is responsible for processing the model's output and displaying it
# to the user in a human readable format.
#

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model import predict
import numpy as np

from typing import Union, Dict

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


def output(imageMatrix: np.ndarray) -> Dict[str, Union[str, np.ndarray]]:
    """
    Processes the model's output and displays it to the user in a human readable format.

    Parameters:
    imageMatrix (np.ndarray): The matrix representation of the image

    Returns:
    Dict[str, Union[str, np.ndarray]]: A dictionary containing the predicted label and the confidence matrix
    """

    prediction = predict(imageMatrix)

    return {
        "predictedLabel": LABELS[np.argmax(prediction)],
        "confidenceMatrix": prediction,
    }
