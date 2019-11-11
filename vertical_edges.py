import numpy as np

import utility
from mean_blur import mean_blur


def build_vertical_edges_kernel():
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    return kernel

def vertical_edges_filter(image):
    kernel = build_vertical_edges_kernel()
    vertical_edges_img = utility.apply_kernel(image, kernel)
    return vertical_edges_img

def main():
    kernel_size = 21
    image_path = '/home/uros/Desktop/cat.jpg'
    
    image = utility.load_image(image_path, grayscale=True)

    image = mean_blur(image, 15)

    vertical_edges_img = vertical_edges_filter(image)
    vertical_edges_img_path = '/home/uros/Desktop/vertical_edge_cat_blur15.jpg'.format(kernel_size)
    utility.save_image(vertical_edges_img_path, vertical_edges_img)

if __name__ == '__main__':
    main()
