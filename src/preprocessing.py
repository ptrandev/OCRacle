#
# Image Preprocessing Module
#
# This module is responsible for processing the input image and preparing it for
# the output module.
#

from PIL import Image
import numpy as np


def preprocessing(image: Image) -> np.ndarray:
    """
    Preprocesses the input image and returns the processed image.

    Parameters:
    image (PIL.Image): The input image

    Returns:
    PIL.Image: The processed image
    """

    return normalization(bicubicInterpolation(image))


def bicubicInterpolation(image: Image, width: int = 28, height: int = 28) -> Image:
    """
    Resize the input image using bicubic interpolation.

    Parameters:
    image (PIL.Image): The input image
    width (int): The width of the output image
    height (int): The height of the output image

    Returns:
    PIL.Image: The resized image
    """

    # resize the image using bicubic interpolation
    return image.resize((width, height), Image.BICUBIC)


def normalization(image: Image) -> np.ndarray:
    """
    Normalize the input image to have pixel values between 0 and 1.

    Parameters:
    image (PIL.Image): The input image

    Returns:
    list[list[int]]: The normalized image
    """

    image = image.convert('L')

    # turn the image into a list of pixel values
    imageMatrix = [[image.getpixel((x, y)) for x in range(image.width)] for y in range(image.height)]

    # normalize the image to have pixel values between 0 and 1
    imageMatrix = [[pixel / 255.0 for pixel in row] for row in imageMatrix]

    # turn into numpy array
    imageMatrix = np.array(imageMatrix).reshape(1, 28, 28)

    return imageMatrix