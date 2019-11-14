import numpy as np
from PIL import Image


def imopen(img_path):
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    return img

def imwrite(img_path, img):
    img = ((img - img.min()) / (img.max() - img.min())) * 255.
    img = Image.fromarray(img.astype(np.uint8))
    img.save(img_path)

def rgb2gray(rgb, conv_ratio=None):
    if conv_ratio is None:
        conv_ratio = np.array([0.2989, 0.5870, 0.1140])

    return np.dot(rgb, conv_ratio)

def imblur_mean(img, filt_shape):
    filt = mean_blur_filter(filt_shape)
    return filter(img, filt)

def imblur_gaussian():
    pass

def imblur_median():
    pass

def imresize(img, height, width, inter='nn'):
    assert inter in ['nn', 'bilinear', 'bicubic'], 'Invalid interpolation'

    # extract old image dimensions
    h, w = img.shape

    x_ratio = height / h
    y_ratio = width / h

    resized_row_indices = np.arange(height).reshape((height, 1)) * np.ones((height, width))         # [height, width]
    resized_column_indices = np.transpose(np.arange(width).reshape((width, 1)) * np.ones((width, height)))  # [height, width]

    resized_image_indices = np.stack([resized_row_indices, resized_column_indices], axis=-1)    # [height, width, 2]
    resized_image_indices = resized_image_indices.reshape((-1, 2))              # [height * width, 2]

    inverse_transformation = np.array([[1 / x_ratio, 0],
                                       [0, 1 / y_ratio]])       # [2, 2]

    projected_pixel_positions = resized_image_indices @ inverse_transformation  #[height * width, 2]


    original_row_indices = np.arange(h).reshape((h, 1)) * np.ones((h, w))         # [h, w]
    original_column_indices = np.transpose(np.arange(w).reshape((w, 1)) * np.ones((w, h)))  # [h, w]

    original_image_indices = np.stack([original_row_indices, original_column_indices], axis=-1)    # [h, w, 2]
    original_image_indices = original_image_indices.reshape((-1, 2))              # [h * w, 2]

    axis_difference = np.expand_dims(original_image_indices, 0) - np.expand_dims(projected_pixel_positions, 1) # [height * width, h * w, 2]

    eucleadean_difference = np.sqrt(axis_difference[:, :, 0]**2 + axis_difference[:, :, 1]**2)  # [height * width, h * w]

    closest_points = np.argmin(eucleadean_difference, axis=-1)  # [height * width]

    wanted_indices = original_image_indices[closest_points.astype(np.int32)].astype(np.int32)  # [height * width, 2]

    resized_image = img[wanted_indices[:, 0], wanted_indices[:, 1]]

    resized_image = resized_image.reshape((height, width))

    if inter == 'nn':
        pass

    elif inter == 'bilinear':
        pass

    elif inter == 'bicubic':
        pass

    return resized_image

def imaffine():
    pass

def filter(img, filt):
    grayscale = len(img.shape) == 2

    if grayscale:
        h, w = image.shape
        image = np.expand_dims(image, -1)
        c = 1
    else:
        h, w, c = image.shape

    kernel_size = filt.shape[0]
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

def sobel_edge_det(img, blur_filt_shape):
    img = imblur_mean(img, blur_filt_shape)

    filt_ver = sobel_filter('ver')
    edges_ver = filter(img, filt_ver)

    filt_hor = sobel_filter('hor')
    edges_hor = filter(img, filt_hor)

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

def fit_affine(a, b):
    pass

def affine_transform():
    pass

# TODO: copy syntax of cv2
def threshold():
    pass