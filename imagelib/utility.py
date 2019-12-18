import numpy as np
from PIL import Image

#TODO: form exception in case of an error.
def imopen(img_path):
    """
    Given filepath to an image, this function loads it as an numpy.ndarray
    of type float32.

    Args:
        img_path(string): Filepath to an image.
    Ret:
        Image as an numpy.ndarray.
    """
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)

    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = img[:, :, :-1]

    return img


def imwrite(img_path, img):
    """
    Given filepath and an image in numpy.ndarray format,
    this functions stores it.

    Args:
        img_path(string): Filepath to a desired location.
        img(numpy.ndarray): RGB or grayscale image.
    """
    img = ((img - img.min()) / (img.max() - img.min())) * 255.
    img = Image.fromarray(img.astype(np.uint8))
    img.save(img_path)


# TODO: Extend this to cover RGBA.
def rgb2gray(rgb, conv_ratio=None):
    """
    Converts rgb image to grayscale.

    Args:
        rgb(numpy.ndarray): RGB image with shape [<height>, <width>, 3].
        conv_ration(numpy.ndarray): Denotes how much green, red or blue affect
            intensity grayscale image. Shape [1, 3]. Form [red_contrib, green_contrib, blue_contrib].
    Ret:
        Grayscale image as an numpy.ndarray of shape [<height>, <width>].
    """
    if len(rgb.shape) == 2:
        return rgb

    if conv_ratio is None:
        conv_ratio = np.array([0.2989, 0.5870, 0.1140])

    return np.dot(rgb, conv_ratio)


#TODO
def rgb2hsv():
    pass


#TODO
def hsv2rgb():
    pass