#
# Data Module
#
# This module is responsible for fetching and preprocessing the EMNIST dataset.
#

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.libraries.emnist import extract_test_samples, extract_training_samples


def extractTestSamples():
    """
    Load the EMNIST test dataset and preprocess the images and labels.

    Returns:
    tuple: The preprocessed images and labels
    - images (np.ndarray): The preprocessed images
    - labels (np.ndarray): The preprocessed labels
    """

    test_images, test_labels = extract_test_samples("letters")

    return processSamples(test_images, test_labels)


def extractTrainingSamples() -> tuple:
    """
    Load the EMNIST training dataset and preprocess the images and labels.

    Returns:
    tuple: The preprocessed images and labels
    - images (np.ndarray): The preprocessed images
    - labels (np.ndarray): The preprocessed labels
    """

    images, labels = extract_training_samples("letters")

    return processSamples(images, labels)


def processSamples(images, labels) -> tuple:
    """
    Preprocess the images and labels.
    - Normalize the images to a pixel range of 0 to 1
    - Scale the labels to a range of 0 to 25

    Parameters:
    images (np.ndarray): The images to preprocess
    labels (np.ndarray): The labels to preprocess

    Returns:
    tuple: The preprocessed images and labels
    """

    # Normalize the images to a pixel range of 0 to 1
    images = images / 255.0

    # scale labels to 0-25
    labels = labels - 1

    return images, labels
