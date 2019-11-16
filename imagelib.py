import numpy as np
from PIL import Image


def imopen(img_path):
    """
    Given filepath to an image, loads it as an numpy.ndarray of type float32.
    """
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    return img

def imwrite(img_path, img):
    """
    Given filepath and an image in numpy.ndarray format, this functions stores it.

    img_path(string): Filepath to a desired location.
    img(numpy.ndarray): RGB or grayscale image.
    """
    img = ((img - img.min()) / (img.max() - img.min())) * 255.
    img = Image.fromarray(img.astype(np.uint8))
    img.save(img_path)

# TODO: Extend this to cover RGBA.
def rgb2gray(rgb, conv_ratio=None):
    """
    Converts rgb image to grayscale.

    rgb(numpy.ndarray): RGB image with shape [<height>, <width>, 3].
    conv_ration(numpy.ndarray): Denotes how much green, red or blue affect
        intensity grayscale image. Shape [1, 3]. Form [red_contrib, green_contrib, blue_contrib].

    """
    if conv_ratio is None:
        conv_ratio = np.array([0.2989, 0.5870, 0.1140])

    return np.dot(rgb, conv_ratio)

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

def imrotate():
    pass

def imaffine(img, transformation, inter='nn'):
    assert transformation.shape == (2, 3), 'imagelib.imaffine: Invalid transformation matrix'
    assert inter in ['nn', 'bilinear', 'bicubic'], 'imagelib.imaffine: Invalid interpolation type'

    # extract affine transformation parameters
    affine_A = transformation[:, :2]
    affine_b = transformation[:, -1:]

    img_h, img_w = img.shape

    # corners of the image
    A = np.array([-img_w/2, img_h/2], dtype=np.float32).reshape((2, 1))
    B = np.array([-img_w/2, -img_h/2], dtype=np.float32).reshape((2, 1))
    C = np.array([img_w/2, -img_h/2], dtype=np.float32).reshape((2, 1))
    D = np.array([img_w/2, img_h/2], dtype=np.float32).reshape((2, 1))

    # corners of the transformed image
    proj_A = affine_A @ A + affine_b
    proj_B = affine_A @ B + affine_b
    proj_C = affine_A @ C + affine_b
    proj_D = affine_A @ D + affine_b

    x_min = np.min([proj_A[0, :], proj_B[0, :], proj_C[0, :], proj_D[0, :]])
    x_max = np.max([proj_A[0, :], proj_B[0, :], proj_C[0, :], proj_D[0, :]])
    
    y_min = np.min([proj_A[1, :], proj_B[1, :], proj_C[1, :], proj_D[1, :]])
    y_max = np.max([proj_A[1, :], proj_B[1, :], proj_C[1, :], proj_D[1, :]])

    # finding dimensions of outter transformed image
    new_img_h = np.int32(y_max - y_min)
    new_img_w = np.int32(x_max - x_min)

    new_img = np.zeros([new_img_h, new_img_w])

    # coordinates of points in new image
    new_img_coords_y, new_img_coords_x = np.meshgrid(np.arange(new_img_w, dtype=np.float32), np.arange(new_img_h, dtype=np.float32))
    new_img_coords = np.stack([new_img_coords_x, new_img_coords_y], axis=-1)    # GOOD!
    new_img_coords = new_img_coords.reshape((-1, 2))
    new_img_coords = np.transpose(new_img_coords)   

    image_coord_sys_new_img_coords = new_img_coords[:, :]
    new_img_coords +=  0.5

    # transfer those coordinates from image coordinate system to eucleadean coordinate system
    rot_mat = np.array([[ 0, 1], 
                        [-1, 0]], dtype=np.float32)
    trans_vect = np.array([-new_img_w/2, new_img_h/2], dtype=np.float32).reshape((2, 1))

    new_img_coords = rot_mat @ new_img_coords + trans_vect

    # calculate parameters of inverse affine transformation
    inv_affine_A = np.linalg.inv(affine_A)
    inv_affine_b = -inv_affine_A @ affine_b

    # coordinates of point in new image projected onto the old image
    projected_new_coords = inv_affine_A @ new_img_coords + inv_affine_b

    # transfer those coordinates to image coordinate system of original image
    trans_vect = np.array([img_w/2, -img_h/2], dtype=np.float32).reshape((2, 1))
    rot_mat = np.array([[0, -1], 
                        [1,  0]], dtype=np.float32)

    projected_new_coords = rot_mat @ (projected_new_coords + trans_vect)
    
    # coordinates of points in old image
    old_img_coords_y, old_img_coords_x = np.meshgrid(np.arange(img_w, dtype=np.float32), np.arange(img_h, dtype=np.float32))
    old_img_coords = np.stack([old_img_coords_x, old_img_coords_y], axis=-1)    # GOOD!
    old_img_coords = old_img_coords.reshape((-1, 2))
    old_img_coords = np.transpose(old_img_coords)  
    old_img_coords +=  0.5 

    projected_new_coords = np.transpose(projected_new_coords)
    new_img_coords = np.transpose(new_img_coords)
    old_img_coords = np.transpose(old_img_coords)
    image_coord_sys_new_img_coords = np.transpose(image_coord_sys_new_img_coords)

    if inter == 'nn':
        for idx, (proj_coord, coord) in enumerate(zip(projected_new_coords, image_coord_sys_new_img_coords)):

            if proj_coord[0] < 0 or proj_coord[0] > img_h or\
                proj_coord[1] < 0 or proj_coord[1] > img_w:

                coord = coord.astype(np.int32)
                new_img[coord[0], coord[1]] = 255
                continue

            axis_differences = old_img_coords - proj_coord
            differences = np.sqrt(axis_differences[:, 0]**2 + axis_differences[:, 1]**2)
            closest_point = np.argmin(differences)

            coord = coord.astype(np.int32)

            old_img_coord = old_img_coords[closest_point, :].astype(np.int32)

            new_img[coord[0], coord[1]] = img[old_img_coord[0], old_img_coord[1]]

    elif inter == 'bilinear':
        pass

    elif inter == 'bicubic':
        pass

    return new_img

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

def rgb2hsv():
    pass

def hsv2rgb():
    pass

def is_rgb(img):
    pass

def is_rgba(img):
    pass