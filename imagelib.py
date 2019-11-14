import numpy as np
from PIL import Image


def imopen(img_path):
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    return img

def imwrite(img_path, img):
    img = ((img - img.min()) / (img.max() - img.min())) * 255.
    img = Image.fromarray(img.astype(np.uint8))
    img.save(img_path)

def rgb2gray(rgb, conv_ratio=None):
    if conv_ratio is None:
        conv_ratio = np.array([0.2989, 0.5870, 0.1140])

    return np.dot(rgb, conv_ratio)

def imblur_mean():
    pass

def imblur_gaussian():
    pass

def imblur_median():
    pass

def imresize():
    pass

def imaffine():
    pass

def filter():
    pass

def canny_edge():
    pass

def histeq():
    pass

def histspec():
    pass

def imsharpen():
    pass

def sobel_filter(type):
    filter = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    if type == 'ver':
        return filter
    elif type == 'hor':
        return np.transpose(filter)
        

def laplassian_filter():
    filter = np.array([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]])
    return filter