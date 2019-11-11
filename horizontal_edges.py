import numpy as np

import utility
from mean_blur import mean_blur


def build_horizontal_edges_kernel():
    kernel = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    return kernel

def horizontal_edges_filter(image):
    kernel = build_horizontal_edges_kernel()
    horizontal_edges_img = utility.apply_kernel(image, kernel)
    return horizontal_edges_img

def main():
    kernel_size = 21
    image_path = '/home/uros/Desktop/cat.jpg'
    
    image = utility.load_image(image_path, grayscale=True)

    # image = mean_blur(image, 15)

    horizontal_edges_img = horizontal_edges_filter(image)
    horizontal_edges_img_path = '/home/uros/Desktop/horizontal_edge_cat_blur.jpg'.format(kernel_size)
    utility.save_image(horizontal_edges_img_path, horizontal_edges_img)

if __name__ == '__main__':
    main()
