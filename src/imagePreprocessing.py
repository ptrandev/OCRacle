#
# Image Preprocessing Module
#
#  This module is responsible for processing the input image and preparing it for
#  the output module.
#

from PIL import Image

def preprocessing(image: Image) -> list[list[int]]:
    """
    Preprocesses the input image and returns the processed image.

    Parameters:
    image (PIL.Image): The input image

    Returns:
    PIL.Image: The processed image
    """

    # resize the image to 28x28 pixels using bicubic interpolation
    image = bicubicInterpolation(image)

    # convert the image to grayscale, from 0 to 255
    image = normalization(image)

    # turn the image into a 28 x 28 matrix
    imageMatrix = [[image.getpixel((x, y)) for x in range(28)] for y in range(28)]

    # return the processed image
    return imageMatrix

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

def normalization(image: Image) -> Image:
    '''
    Normalize the input image to have pixel values between 0 and 255.

    Parameters:
    image (PIL.Image): The input image

    Returns:
    PIL.Image: The normalized image
    '''

    return image.convert("L")