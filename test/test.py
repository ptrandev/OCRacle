import sys
import os
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.input import input
from src.preprocessing import preprocessing

TEST_IMAGES_PATH = "testImages/"


def jpegAcceptanceTest():
    """
    T1: Test reading a valid JPEG image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.jpg")
    image = input(filePath)
    assert image is not None
    assert len(image) == 28
    assert len(image[0]) == 28


def pngAcceptanceTest():
    """
    T2: Test reading a valid PNG image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    # test reading a valid image file
    image = input(filePath)
    assert image is not None
    assert len(image) == 28
    assert len(image[0]) == 28


def nonSupportedFormatRejectionTest():
    """
    T3: Test reading an invalid image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.pdf")

    # test reading an invalid image file
    try:
        input(filePath)
    except Exception as e:
        assert str(e) == f"File {filePath} is not a valid image file"


def imagePreProcessingTest():
    """
    T4: Test image preprocessing
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.jpg")

    # read the filePath into a File object
    image = Image.open(filePath)

    # test preprocessing function
    processedImage = preprocessing(image)

    # ensure that the processed image is not None
    assert processedImage is not None

    # ensure dimensions of the processed image are 28 x 28
    assert len(processedImage) == 28
    assert len(processedImage[0]) == 28

    # ensure that image is in grayscale
    for row in processedImage:
        for pixel in row:
            assert pixel >= 0 and pixel <= 255


if __name__ == "__main__":
    jpegAcceptanceTest()
    pngAcceptanceTest()
    nonSupportedFormatRejectionTest()
    imagePreProcessingTest()
    print("All tests passed!")
