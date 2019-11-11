import numpy as np

import edge_detector
import utility

def strong_edge_detector(sobel_edges_image, sobel_edges_directions):
    h, w = sobel_edges_image.shape

    strong_edges_image = np.zeros(sobel_edges_image.shape)

    dir = None
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if 3*np.pi/8 <= sobel_edges_directions[i][j] < np.pi/2 or\
               -3*np.pi/8 <= sobel_edges_directions[i][j] < -np.pi/8:
                dir = 1
            elif np.pi/8 <= sobel_edges_directions[i][j] < 3*np.pi/8:
                dir = 2
            elif -np.pi/8 <= sobel_edges_directions[i][j] < np.pi/8:
                dir = 3
            else:
                dir = 4

            if dir == 1:
                if sobel_edges_directions[i][j] > sobel_edges_directions[i][j - 1] and\
                    sobel_edges_directions[i][j] > sobel_edges_directions[i][j + 1]:
                    strong_edges_image[i][j] = sobel_edges_image[i][j]
            elif dir == 2:
                if sobel_edges_directions[i][j] > sobel_edges_directions[i - 1][j - 1] and\
                    sobel_edges_directions[i][j] > sobel_edges_directions[i + 1][j + 1]:
                    strong_edges_image[i][j] = sobel_edges_image[i][j]
            elif dir == 3:
                if sobel_edges_directions[i][j] > sobel_edges_directions[i - 1][j] and\
                    sobel_edges_directions[i][j] > sobel_edges_directions[i + 1][j]:
                    strong_edges_image[i][j] = sobel_edges_image[i][j]
            else:
                if sobel_edges_directions[i][j] > sobel_edges_directions[i - 1][j + 1] and\
                    sobel_edges_directions[i][j] > sobel_edges_directions[i + 1][j - 1]:
                    strong_edges_image[i][j] = sobel_edges_image[i][j]

    return strong_edges_image


def apply_dual_threshold(strong_edges_image, high_thresh, low_thresh):
    strong_edges_image[strong_edges_image < low_thresh] = 0.

    return strong_edges_image

def canny_edge_detector(image, mean_blue_kernel_size, high_thresh, low_thresh):
    sobel_edges_image, sobel_edges_directions = edge_detector.edge_detector(image, mean_blue_kernel_size)

    strong_edges_image = strong_edge_detector(sobel_edges_image, sobel_edges_directions)
    canny_edges_image = apply_dual_threshold(strong_edges_image, high_thresh, low_thresh)
    
    return canny_edges_image

def main():
    blur_kernel_size = 15
    image_path = '/home/uros/Desktop/vase.jpg'
    high_thresh = 0.05
    low_thresh = 0.005
    
    image = utility.load_image(image_path, grayscale=True)
    canny_edges_img = canny_edge_detector(image, blur_kernel_size, high_thresh, low_thresh)

    canny_edges_img_path = '/home/uros/Desktop/vase_canny_edge_detector_blur{}.jpg'.format(blur_kernel_size)
    utility.save_image(canny_edges_img_path, canny_edges_img)

if __name__ == '__main__':
    main()
