#
# Prediction Model Module
#
# This module is responsible for loading the trained model and making
# predictions on the input image.
#

import os
import tensorflow as tf

MODEL = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "model.keras")
)


def predict(imageMatrix):
    return MODEL.predict(imageMatrix)
