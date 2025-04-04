#
# Input Format Module
#
# This module is responsible for reading the input file and parsing it into a
# format that can be used by the rest of the program.
#

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
import src.preprocessing as preprocessing
import numpy as np

N_MIN = 28
N_MAX = 2048
M_MIN = 28
M_MAX = 2048


def input(filePath: str) -> np.ndarray:
    """
    Reads the input file and returns a matrix representation of the image.

    Parameters:
    filePath (str): The path to the input file

    Returns:
    np.ndarray: The matrix representation of the image

    Exceptions:
    FileNotFoundError: If the file is not found
    DimensionsError: If the image dimensions are not valid
    FileFormatError: If the file format is not valid
    """

    # Check if the file exists
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"File {filePath} not found")

    # open the image file using PIL
    try:
        image = Image.open(filePath)
    except Exception:
        raise FileFormatError(f"File {filePath} is not a valid image file")

    # ensure that the image is a JPG, JPEG, or PNG file
    if image.format not in ["JPEG", "JPG", "PNG"]:
        raise FileFormatError(f"File {filePath} is not a valid image file")

    # get the dimensions of the image
    width, height = image.size

    # ensure that the image dimensions are valid
    if width < N_MIN or width > N_MAX or height < M_MIN or height > M_MAX:
        raise DimensionsError(f"Image dimensions {width}x{height} are not valid")

    # return the image content
    return preprocessing.preprocessing(image)


# define custom exceptions
class DimensionsError(Exception):
    pass


class FileFormatError(Exception):
    pass
