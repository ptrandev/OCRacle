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
  test_images, test_labels = extract_test_samples("letters")

  return processSamples(test_images, test_labels)


def extractTrainingSamples():
  # Load EMNIST data
  images, labels = extract_training_samples("letters")

  return processSamples(images, labels)


def processSamples(images, labels):
  # Normalize the images to a pixel range of 0 to 1
  images = images / 255.0

  # scale labels to 0-25
  labels = labels - 1

  return images, labels