import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagelib.filter as filt
import imagelib.utility as util

if __name__ == '__main__':
    LOAD_IMG_PATH = '/home/uros/Desktop/workspace/grid.jpg'
    MAIN_EDGES_PATH = '/home/uros/Desktop/workspace/grid_thin_edge.jpg'
    CANNY_PATH = '/home/uros/Desktop/workspace/grid_canny.jpg'
    HOUGH_PATH = '/home/uros/Desktop/workspace/grid_hough.jpg'
    DBG_PATH = '/home/uros/Desktop/workspace/grid_dbg.jpg'
    SOBEL_EDGE = '/home/uros/Desktop/workspace/grid_sobel.jpg'

    img = util.imopen(LOAD_IMG_PATH)
    rgb_img = img[:, :, :]
    img = util.rgb2gray(img)

    edge_img, edge_dir = filt.sobel_edge_det(img, (3, 3))
    util.imwrite(SOBEL_EDGE, edge_img)

    main_edge_img = filt.non_max_supression(edge_img, edge_dir)
    util.imwrite(MAIN_EDGES_PATH, main_edge_img)

    high = 60
    low = 20
    canny_edges_image = filt.dual_threshold(main_edge_img, high, low)
    canny_edges_image = canny_edges_image[5:-5, 5:-5]
    util.imwrite(CANNY_PATH, canny_edges_image)

    hough_edges, hough_img = filt.hough_transform(canny_edges_image, 100, 100, 100)
    util.imwrite(DBG_PATH, hough_img)

    rgb_img = rgb_img[5:-5, 5:-5]

    for x in range(hough_img.shape[0]):
        for y in range(hough_img.shape[1]):
            if hough_img[x, y] == 1:
                rgb_img[x, y, 1] = 255.

    util.imwrite(HOUGH_PATH, rgb_img)

