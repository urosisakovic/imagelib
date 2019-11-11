import numpy as np

import utility
from horizontal_edges import horizontal_edges_filter
from mean_blur import mean_blur
from vertical_edges import vertical_edges_filter


def main():
    blur_kernel_size = 15
    image_path = '/home/uros/Desktop/cat.jpg'
    
    image = utility.load_image(image_path, grayscale=True)

    image = mean_blur(image, blur_kernel_size)

    vertical_edges_img = vertical_edges_filter(image)
    horizontal_edges_img = horizontal_edges_filter(image)

    edges_img = np.sqrt(vertical_edges_img**2 + horizontal_edges_img**2)

    edges_img_path = '/home/uros/Desktop/edge_detector_blur{}.jpg'.format(blur_kernel_size)
    utility.save_image(edges_img_path, edges_img)

if __name__ == '__main__':
    main()
