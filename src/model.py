#
# Prediction Model Module
#
# This module is responsible for loading the trained model and making
# predictions on the input image.
#

import os
import keras

from typing import cast

# Adding an explicit cast to help static type checkers like Pylance
MODEL = cast(
    keras.Model,
    keras.models.load_model(os.path.join(os.path.dirname(__file__), "model.keras")),
)
