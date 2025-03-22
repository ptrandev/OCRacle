import os
import sys
import pytest
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.input import input
from src.preprocessing import preprocessing
from src.model import predict
from src.output import output

TEST_IMAGES_PATH = os.path.join(os.path.dirname(__file__), "testImages/")


def testJpegAcceptance():
    """
    T1: Test reading a valid JPEG image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.jpg")

    image = input(filePath)

    assert image is not None
    assert image.shape == (1, 28, 28)


def testPngAcceptance():
    """
    T2: Test reading a valid PNG image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)

    # Ensure that the image is not None and has the correct shape
    assert image is not None
    assert image.shape == (1, 28, 28)


def testNonSupportedFormatRejection():
    """
    T3: Test reading an invalid image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.pdf")

    # Test reading an invalid image file, this should raise an exception
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

    # ensure that the processed image is not None and has the correct shape
    assert processedImage is not None
    assert processedImage.shape == (1, 28, 28)

    # ensure min and max pixel values are 0 and 1
    assert np.min(processedImage) == 0
    assert np.max(processedImage) == 1


def testCharacterPrediction():
    """
    T5: Test character prediction
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)
    prediction = output(image)

    # Ensure that the prediction is not None and has the correct label
    assert prediction is not None
    assert prediction["predictedLabel"] == "Y"

    # Ensure that the confidence matrix is not None and has the correct shape
    assert prediction["confidenceMatrix"] is not None
    assert prediction["confidenceMatrix"].shape == (1, 27)


def testProbabilityVectorSum():
    """
    T6: Test probability vector sum
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)
    prediction = predict(image)

    # Ensure the sum of the probability vector is 1
    assert np.sum(prediction) == 1


def testProbabilityVectorLength():
    """
    T7: Test probability vector length
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)
    prediction = predict(image)

    # Ensure the shape of the probability vector is correct
    assert prediction.shape == (1, 27)


def humanReadableOutput():
    """
    T8: Test human readable output
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)
    prediction = output(image)

    # Ensure that the human readable output is correct
    assert prediction["predictedLabel"] == "Y"
