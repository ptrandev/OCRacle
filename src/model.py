#
# Prediction Model Module
#
# This module is responsible for loading the trained model and making
# predictions on the input image.
#

import os
import keras
import numpy as np

from typing import cast

# Adding an explicit cast to help static type checkers like Pylance
MODEL = cast(keras.Model, keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "model.keras")
))

def predict(imageMatrix: np.ndarray) -> np.ndarray:
    """
    Make a prediction on the input image matrix using the pre-trained model.

    Parameters:
    imageMatrix (np.ndarray): The input image as a numpy array

    Returns:
    np.ndarray: The predicted output from the model
    """
    return MODEL.predict(imageMatrix, verbose="0")