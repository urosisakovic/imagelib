import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagelib.filter as filt
import imagelib.utility as util

if __name__ == '__main__':
    LOAD_IMG_PATH = '/home/uros/Desktop/workspace/ana.png'
    SOBEL_IMG_PATH = '/home/uros/Desktop/workspace/sobel_edges.jpg'
    MAIN_EDGES_PATH = '/home/uros/Desktop/workspace/main_edges.jpg'
    CANNY_PATH = '/home/uros/Desktop/workspace/canny.jpg'
    DBG_PATH = '/home/uros/Desktop/workspace/dbg.jpg'

    img = util.imopen(LOAD_IMG_PATH)
    img = util.rgb2gray(img)

    edge_img, edge_dir = filt.sobel_edge_det(img, (3, 3))
    main_edge_img = filt.non_max_supression(edge_img, edge_dir)
    util.imwrite(MAIN_EDGES_PATH, main_edge_img)

    high = 10
    low = 0
    canny_edges_image = filt.dual_threshold(main_edge_img, high, low)
    util.imwrite(CANNY_PATH, canny_edges_image)
