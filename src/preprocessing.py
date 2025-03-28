#
# Image Preprocessing Module
#
# This module is responsible for processing the input image and preparing it for
# the output module.
#

from PIL import Image
import numpy as np


def preprocessing(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the input image and returns the processed image.

    Parameters:
    image (Image.Image): The input image

    Returns:
    np.ndarray: The processed image
    """

    return normalization(bicubicInterpolation(image))


def bicubicInterpolation(image: Image.Image, width: int = 28, height: int = 28) -> Image.Image:
    """
    Resize the input image using bicubic interpolation.

    Parameters:
    image (Image.Image): The input image
    width (int): The width of the output image
    height (int): The height of the output image

    Returns:
    Image.Image: The resized image
    """

    # resize the image using bicubic interpolation
    return image.resize((width, height), Image.Resampling.BICUBIC)


def normalization(image: Image.Image) -> np.ndarray:
    """
    Normalize the input image to have pixel values between 0 and 1.

    Parameters:
    image (Image.Image): The input image

    Returns:
    np.ndarray: The normalized image
    """

    image = image.convert("L")

    # Convert image to a numpy array
    imageMatrix = np.asarray(image, dtype=np.float32)

    # Normalize the image to have pixel values between 0 and 1
    imageMatrix /= 255.0

    # turn into numpy array
    imageMatrix = np.array(imageMatrix).reshape(1, 28, 28)

    return imageMatrix
