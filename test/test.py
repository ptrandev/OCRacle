import os
import sys
import pytest
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.input import input
from src.preprocessing import preprocessing

TEST_IMAGES_PATH = "testImages/"

def testJpegAcceptance():
    """
    T1: Test reading a valid JPEG image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.jpg")
    image = input(filePath)
    assert image is not None
    assert len(image) == 28
    assert len(image[0]) == 28


def testPngAcceptance():
    """
    T2: Test reading a valid PNG image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    # test reading a valid image file
    image = input(filePath)
    assert image is not None
    assert len(image) == 28
    assert len(image[0]) == 28


def testNonSupportedFormatRejection():
    """
    T3: Test reading an invalid image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.pdf")

    # Test reading an invalid image file
    with pytest.raises(Exception) as excinfo:
        input(filePath)
    assert str(excinfo.value) == f"File {filePath} is not a valid image file"

def testImagePreProcessing():
    """
    T4: Test image preprocessing
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.jpg")

    # Read the filePath into a File object
    image = Image.open(filePath)

    # Test preprocessing function
    processedImage = preprocessing(image)

    # Ensure that the processed image is not None
    assert processedImage is not None

    # Ensure dimensions of the processed image are 28 x 28
    assert len(processedImage) == 28
    assert len(processedImage[0]) == 28

    # Ensure that image is in grayscale
    for row in processedImage:
        for pixel in row:
            assert 0 <= pixel <= 255
