import os
import sys
import pytest
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.input import input
from src.preprocessing import preprocessing
from src.model import MODEL
from src.output import output
from src.accuracy import evaluate
from src.data import extractTestSamples, extractTrainingSamples

TEST_IMAGES_PATH = os.path.join(os.path.dirname(__file__), "testImages/")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../src/model.keras")


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
    assert prediction["confidenceMatrix"].shape == (1, 26)


def testProbabilityVectorSum():
    """
    T6: Test probability vector sum
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)
    prediction = MODEL.predict(image, verbose="0")

    # Ensure the sum of the probability vector is 1, with some tolerance
    assert np.sum(prediction) == pytest.approx(1.0, abs=1e-3)


def testProbabilityVectorLength():
    """
    T7: Test probability vector length
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)
    prediction = MODEL.predict(image, verbose="0")

    # Ensure the shape of the probability vector is correct
    assert prediction.shape == (1, 26)


def testHumanReadableOutput():
    """
    T8: Test human readable output
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)
    prediction = output(image)

    # Ensure that the human readable output is correct
    assert prediction["predictedLabel"] == "Y"


def testAccuracyMeasurement():
    """
    T9: Test accuracy measurement
    """

    evaluation = evaluate()

    # Ensure that the evaluation exists
    assert evaluation is not None
    # ensure accuracy is better than the previous OAR model
    assert evaluation["accuracy"] > 0.674

def testLoadTrainSubset():
    """
    T15: Test load train subset
    """

    images, labels = extractTestSamples()

    assert images is not None
    assert labels is not None

    # ensure images and length are 20,800
    assert len(images) == 20800
    assert len(labels) == 20800

    # ensure images is the correct shape
    assert images.shape == (len(images), 28, 28)

    # ensure values of each image is 0-1 inclusive
    assert np.min(images) == 0
    assert np.max(images) == 1

    # ensure label values are 0-25 inclusive
    assert np.min(labels) == 0
    assert np.max(labels) == 25

def testLoadTestSubset():
    """
    T16: Test load train subset
    """

    images, labels = extractTrainingSamples()

    assert images is not None
    assert labels is not None

    # ensure images and length are 124800
    assert len(images) == 124800
    assert len(labels) == 124800

    # ensure images is the correct shape
    assert images.shape == (len(images), 28, 28)

    # ensure values of each image is 0-1 inclusive
    assert np.min(images) == 0
    assert np.max(images) == 1

    # ensure label values are 0-25 inclusive
    assert np.min(labels) == 0
    assert np.max(labels) == 25