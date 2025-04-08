#
# Image Preprocessing Module
#
# This module is responsible for processing the input image and preparing it for
# the output module.
#

from PIL import Image
import numpy as np

WIDTH = 28
HEIGHT = 28

def preprocessing(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the input image and returns the processed image.

    Parameters:
    image (Image.Image): The input image

    Returns:
    np.ndarray: The processed image
    """

    # Resize the image to 28x28 pixels using bicubic interpolation
    image = image.resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)

    # Convert image to grayscale
    image = image.convert("L")

    # Convert image to a numpy array
    imageMatrix = np.asarray(image, dtype=np.float32)

    # Normalize the image to have pixel values between 0 and 1
    imageMatrix /= 255.0

    # turn into numpy array
    imageMatrix = np.array(imageMatrix).reshape(1, 28, 28)

    return imageMatrix