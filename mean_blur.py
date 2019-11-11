import numpy as np

import utility


def build_mean_blur_kernel(kernel_size):
    kernel = np.ones([kernel_size, kernel_size])
    kernel = np.stack([kernel, kernel, kernel], axis=-1)
    return kernel

def mean_blur(image, kernel_size):
    kernel = build_mean_blur_kernel(kernel_size)
    blurred_image = utility.apply_kernel(image, kernel)
    return blurred_image

def main():
    kernel_size = 21
    image_path = '/home/uros/Desktop/cat.jpg'
    
    image = utility.load_image(image_path)
    blurred_image = mean_blur(image, kernel_size)
    blurred_image_path = '/home/uros/Desktop/blurred_cat_{}.jpg'.format(kernel_size)
    utility.save_image(blurred_image_path, blurred_image)

if __name__ == '__main__':
    main()
