import numpy as np
from scipy.signal import convolve2d

import filter as fil
import utility as utils


def harris_corner_detection(img, patch_size, threshold_eigen, nms_dist):
    assert len(img.shape) == 2, "Image must be grayscale."

    dx_filter = fil.sobel_filter('hor')
    dx = convolve2d(img, dx_filter)
    dx = np.abs(dx)
    dx = utils.scale(dx[1:-1, 1:-1])
  
    dy_filter = fil.sobel_filter('ver')
    dy = convolve2d(img, dy_filter)
    dy = np.abs(dy)
    dy = utils.scale(dy[1:-1, 1:-1])

    H, W = img.shape
    patch_height, patch_width = patch_size

    feature_score = [[0 for _ in range(W // 10)] for _ in range(H // 10)]
    features = [[0 for _ in range(W // 10)] for _ in range(H // 10)]

    for i in range(patch_height - 1, H):
        print(i)

        for j in range(W - patch_width + 1):

            harris_matrix = np.zeros((2, 2))

            harris_matrix[0, 0] = np.sum(dx[i-patch_height+1 : i, j : j+patch_width-1]**2)
            harris_matrix[0, 1] = np.sum(dx[i-patch_height+1 : i, j : j+patch_width-1] * dy[i-patch_height+1 : i, j : j+patch_width-1])
            harris_matrix[1, 0] = np.sum(dx[i-patch_height+1 : i, j : j+patch_width-1] * dy[i-patch_height+1 : i, j : j+patch_width-1])
            harris_matrix[1, 1] = np.sum(dy[i-patch_height+1 : i, j : j+patch_width-1]**2)

            eigenvalues = np.linalg.eigvals(harris_matrix)
            min_eig = np.min(eigenvalues)

            if min_eig > threshold_eigen:
                if min_eig > feature_score[i // 10][j // 10]:
                    print("i // 10: ", i // 10)
                    print("j // 10: ", j // 10)

                    feature_score[i // 10][j // 10] = min_eig
                    features[i // 10][j // 10] = (i, j)
                    
    final_features = []

    for i in range(H // 10):
        for j in range(W // 10):
            if feature_score[i][j] > 0:
                final_features.append(features[i][j])

    return final_features


def main():
    LOAD_IMG_PATH = "/home/uros/Desktop/workspace/ana.png"

    img = utils.imopen(LOAD_IMG_PATH)
    img = utils.rgb2gray(img)

    print("Image shape: ", img.shape)

    features = harris_corner_detection(img, (7, 7), 7500, 10)

    print("Feature count: ", len(features))

    for feature in features:
        x, y = feature

        for i in range(7):
            for j in range(7):
                img[x-i, y+j] = 255

    utils.imwrite("/home/uros/Desktop/workspace/features.png", img)


if __name__ == '__main__':
    main()
