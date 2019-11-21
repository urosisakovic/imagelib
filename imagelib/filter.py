import numpy as np


def imsharpen():
    """
    Doc string.
    """
    pass


def sobel_filter(type):
    """
    Doc string.
    """
    assert type in ['ver', 'hor'], 'imagelib.sobel_filter: Invalid filter type'
    if type == 'ver':
        return np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    elif type == 'hor':
        return np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
        

def laplassian_filter():
    """
    Doc string.
    """
    filter = np.array([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]])
    return filter


def mean_blur_filter(filt_shape):
    """
    Doc string.
    """
    return np.ones(filt_shape) / np.prod(filt_shape)


def imblur_mean(img, filt_shape):
    """
    Mean blur of an image.
    """
    filt = mean_blur_filter(filt_shape)
    return convolve(img, filt)


def imblur_gaussian():
    """
    Doc string.
    """
    pass


def imblur_median():
    """
    Doc string.
    """
    pass


def convolve(img, filt):
    """
    img(numpy.ndarray): grayscale image
    filt(numpy.ndarray): 2D filter
    """ 
    assert len(img.shape) == 2, 'imagelib.convolve: Invalid image format'
    assert len(filt.shape) == 2, 'imagelib.convolve: Invalid filter format'
    assert np.prod(filt.shape) % 2 == 1, 'imagelib.convolve: Invalid filter format'

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
            # image patch of size [<padding_h>, <padding_w>] centered in [i, j].
            image_patch = padded_img[i : 2*padding_h+i+1, j : 2*padding_w+j+1]

            # apply filter to the patch
            filtered_image[i, j] = np.sum(filt * image_patch)

    return filtered_image


def sobel_edge_det(img, blur_filt_shape):
    """
    Doc string.
    """
    img = imblur_mean(img, blur_filt_shape)

    filt_ver = sobel_filter('ver')
    edges_ver = convolve(img, filt_ver)

    filt_hor = sobel_filter('hor')
    edges_hor = convolve(img, filt_hor)

    edges_img = np.sqrt(edges_ver**2 + edges_hor**2)
    edges_img = edges_img / np.max(edges_img) * 255.
    edges_dir = np.arctan(edges_hor / edges_ver)

    return edges_img, edges_dir


def non_max_supression(edges, gradient):
    """
    Doc string.
    """
    h, w = edges.shape
        
    Z = np.zeros(edges.shape)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            p, q = 255, 255

            if 3*np.pi/8 <= gradient[i][j] <= np.pi/2 or\
                -np.pi/2 <= gradient[i][j] < -3*np.pi/8:
                p = edges[i, j-1]
                q = edges[i, j+1]
            elif np.pi/8 <= gradient[i][j] < 3*np.pi/8:
                p = edges[i-1, j+1]
                q = edges[i+1, j-1]
            elif -np.pi/8 <= gradient[i][j] < np.pi/8:
                p = edges[i, j-1]
                q = edges[i, j+1]
            elif -3*np.pi/8 <= gradient[i][j] < -np.pi/8:
                p = edges[i-1, j-1]
                q = edges[i+1, j+1]

            if edges[i, j] >= p and edges[i, j] >= q:
                Z[i, j] = edges[i, j]

    return Z


def dual_threshold(edges, high, low):
    """
    Doc string.
    """
    edges[edges < low] = 0.
    edges[edges >= high] = high

    threshold = np.zeros(edges.shape)
    threshold[edges > high] = 1.

    h, w = edges.shape

    def dfs(i, j):
        threshold[i][j] = 1.

        stack = []
        stack.append((i, j))

        while not len(stack) == 0:
            u, v = stack[-1]
            stack = stack[:-1]

            threshold[u][v] = 1.

            for x in range(-1, 2):
                for y in range(-1, 2):
                    newU = u + x
                    newV = v + y

                    if newU >= h or newU < 0 or newV >= w or newV < 0:
                        continue

                    if threshold[newU][newV] == 1:
                        continue

                    if low <= edges[newU][newV] <= high:
                        stack.append((newU, newV))

    for i in range(h):
        for j in range(w):
            if threshold[i, j] != 1 and edges[i, j] >= high:
                dfs(i, j)

    edges =  edges * threshold
    edges[edges > 0] = 255.

    return edges


def canny_edge(img, blur_filt_shape, high, low):
    """
    Doc string.
    """
    edges, edges_dir = sobel_edge_det(img, blur_filt_shape)
    edges = non_max_supression(edges, edges_dir)
    canny_edges = dual_threshold(edges, high, low)
    
    return canny_edges


def histeq():
    """
    Doc string.
    """
    pass


def histspec():
    """
    Doc string.
    """
    pass


# TODO: copy syntax of cv2
def threshold():
    """
    Doc string.
    """
    pass


def hough_transform(edge, h, w, t):
    """
    Performs Hough transform.

    Args:
        edge(numpy.ndarray): edge[x, y] = 1 if pixel [x, y] contains an edge,
            otherwise it is 0.
        h(int): Number of bins with respect to the distance of the line from
            top left corner. 
        w(int): Number of bins with respect to the angle of the normal from
            top left corner onto the line.
        t(int): Number of necessery votes in order to declare an existing line.
    Ret:
        List of tuples (rho, theta), one for every declared line.
    """
    
    voting_grid = [[0 for _ in range(w)] for _ in range(h)]

    d_theta = np.pi / w
    d_rho = np.hypot(edge.shape[0], edge.shape[1]) / h

    def vote(x, y):    
        for i in range(w):
            theta = d_theta * i
            rho = np.cos(theta) * x + np.sin(theta) * y
            j = int(rho / d_rho)

            voting_grid[i][j] += 1

    for x in range(edge.shape[0]):
        for y in range(edge.shape[1]):
            if edge[x, y]:
                vote(x, y)

    lines = []

    for i in range(h):
        for j in range(w):
            if (voting_grid[i][j] >= t):
                lines.append((i * d_theta, d_rho * j))

    line_img = np.zeros(edge.shape, dtype=np.int32)
    for (theta, rho) in lines:
        for x in range(edge.shape[0]):
            y = int((rho - np.cos(theta) * x) / np.sin(theta))
            
            if 0 <= y < edge.shape[1]:
                line_img[x-1:x+1, y-1:y+1] = 1

        for y in range(edge.shape[1]):
            x = int((rho - np.sin(theta) * y) / np.cos(theta))
            
            if 0 <= x < edge.shape[0]:
                line_img[x-1:x+1, y-1:y+1] = 1

    return lines, line_img

