import numpy as np
from PIL import Image


def rgb_to_grayscale(rgb_image):
    conversion_ratio = np.array([0.2989, 0.5870, 0.1140])
    grayscale_image = np.dot(rgb_image, conversion_ratio)
    return grayscale_image

def load_image(image_path, grayscale=None):
    image = Image.open(image_path)
    image = np.array(image)

    if grayscale is None:
        return image
    elif grayscale:
        if (len(image.shape) == 2):
            return image
        else:
            return rgb_to_grayscale(image)
    else:
        assert len(image.shape) == 3, 'Image is saved as grayscale.'
        return image

def save_image(image_path, image):
    image = Image.fromarray(image.astype(np.uint8))
    image.save(image_path)

def apply_kernel(image, kernel):
    grayscale = len(image.shape) == 2

    if grayscale:
        h, w = image.shape
        image = np.expand_dims(image, -1)
        c = 1
    else:
        h, w, c = image.shape

    kernel_size = kernel.shape[0]
    if (len(kernel.shape) == 2):
        kernel = np.expand_dims(kernel, -1)

    padding_size = (kernel_size - 1) // 2
    padded_image = np.zeros([h + 2*padding_size, w + 2*padding_size, c])
    padded_image[padding_size:-padding_size, padding_size:-padding_size, :] = image

    processed_image = np.zeros(image.shape)

    for i in range(h):
        for j in range(w):
            for k in range(c):
                processed_image[i, j, k] = np.mean(kernel[:, :, k] * 
                                               padded_image[i : 2*padding_size+i+1, j : 2*padding_size+j+1, k]
                                            )

    processed_image = np.squeeze(processed_image, -1)
    return processed_image
