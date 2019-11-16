import numpy as np


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

def mean_blur_filter(filt_shape):
    return np.ones(filt_shape) / np.prod(filt_shape)

def imblur_mean(img, filt_shape):
    """
    Mean blur of an image.
    """
    filt = mean_blur_filter(filt_shape)
    return filter2(img, filt)

def imblur_gaussian():
    pass

def imblur_median():
    pass



def filter2(img, filt):
    """
    img(numpy.ndarray): grayscale image
    filt(numpy.ndarray): 2D filter
    """ 
    assert len(img.shape) == 2, 'imagelib.filter2: Invalid image format'
    assert len(filt.shape) == 2, 'imagelib.filter2: Invalid filter format'
    assert np.prod(filt.shape) % 2 == 1, 'imagelib.filter2: Invalid filter format'

    img_h, img_w = img.shape
    filt_h, filt_w = filt.shape

    padding_w = (filt_w - 1) // 2
    padding_h = (filt_h - 1) // 2

    vertical_padding = np.zeros((img_h, padding_w))
    horizontal_padding = np.zeros((padding_h, img_w + 2 * padding_w))

    padded_img = np.concatenate([vertical_padding, img, vertical_padding], axis=-1)
    padded_img = np.concatenate([horizontal_padding, padded_img, horizontal_padding], axis=0)

    filtered_image = np.zeros(img.shape)

    for i in range(img_h):
        for j in range(img_w):
            filtered_image[i, j] = np.sum(filt * padded_img[i : 2*vertical_padding+i+1, j : 2*horizontal_padding+j+1])

    return filtered_image

def sobel_edge_det(img, blur_filt_shape):
    img = imblur_mean(img, blur_filt_shape)

    filt_ver = sobel_filter('ver')
    edges_ver = filter2(img, filt_ver)

    filt_hor = sobel_filter('hor')
    edges_hor = filter2(img, filt_hor)

    edges_img = np.sqrt(edges_ver**2 + edges_hor**2)
    edges_dir = np.arctan(edges_hor / edges_ver)

    return edges_img, edges_dir

def main_edges(sobel_edges_image, sobel_edges_directions):
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

def dual_threshold(strong_edges_image, high_thresh, low_thresh):
    strong_edges_image[strong_edges_image < low_thresh] = 0.

    thresholded_image = np.zeros(strong_edges_image.shape)

    thresholded_image[strong_edges_image > high_thresh] = 1.

    h, w = strong_edges_image.shape

    def dfs(i, j):
        thresholded_image[i][j] = 1.

        stack = []
        stack.append((i, j))

        while not len(stack) == 0:
            u, v = stack[-1]
            stack = stack[:-1]

            thresholded_image[u][v] = 1.

            for x in range(-1, 2):
                for y in range(-1, 2):
                    newU = u + x
                    newV = v + y

                    if newU >= h or newU < 0 or newV >= w or newV < 0:
                        continue

                    if thresholded_image[newU][newV] == 1:
                        continue

                    if low_thresh <= strong_edges_image[newU][newV] <= high_thresh:
                        stack.append((newU, newV))

    for i in range(h):
        for j in range(w):
            if thresholded_image[i][j] == 1. and low_thresh <= strong_edges_image[i][j] <= high_thresh:
                dfs(i, j)

    return strong_edges_image * thresholded_image

def canny_edge(img, blur_filt_shape, high, low):
    sobel_edges_image, sobel_edges_directions = sobel_edge_det(img, blur_filt_shape)

    strong_edges_image = main_edges(sobel_edges_image, sobel_edges_directions)
    canny_edges_image = dual_threshold(strong_edges_image, high, low)
    
    return canny_edges_image, strong_edges_image

def histeq():
    pass

def histspec():
    pass

# TODO: copy syntax of cv2
def threshold():
    pass