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
from src.train import train

TEST_IMAGES_PATH = os.path.join(os.path.dirname(__file__), "testImages/")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../src/model.keras")


def testJpegAcceptance():
    """
    T1: Test reading a valid JPEG image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.jpg")

    image = input(filePath)

    assert image is not None

    # image should be 28x28
    assert image.shape == (1, 28, 28)

    # pixel values are between 0 and 1 inclusive
    assert np.min(image) == 0
    assert np.max(image) == 1


def testPngAcceptance():
    """
    T2: Test reading a valid PNG image file
    """
    filePath = os.path.join(TEST_IMAGES_PATH, "Y.png")

    image = input(filePath)

    assert image is not None

    # image should be 28x28
    assert image.shape == (1, 28, 28)

    # pixel values are between 0 and 1 inclusive
    assert np.min(image) == 0
    assert np.max(image) == 1


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

    assert processedImage is not None

    # ensure that the processed image is 28x28
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

    assert prediction is not None

    # ensure predicted label is correct
    assert prediction["predictedLabel"] == "Y"

    assert prediction["confidenceMatrix"] is not None

    # Ensure that the confidence matrix is 1x26
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

    # Ensure the shape of the probability vector is 1x26
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

    assert evaluation is not None

    # ensure overall accuracy is better than the previous OAR model
    assert evaluation["accuracy"] > 0.674


def testModelTraining():
    """
    T14: Test model training
    """

    # Test the training function
    train()

    # Ensure the model is saved to the correct path
    assert os.path.exists(MODEL_PATH)

    # Load the model and check if it is loaded correctly
    loaded_model = MODEL.load_model(MODEL_PATH)
    assert loaded_model is not None

    # Ensure that the loaded model takes in 28x28 input and produces 1x26 output
    assert loaded_model.input_shape == (None, 28, 28)
    assert loaded_model.output_shape == (None, 26)

def testLoadTrainSubset():
    """
    T15: Test load train subset
    """

    images, labels = extractTestSamples()

    assert images is not None
    assert labels is not None

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
