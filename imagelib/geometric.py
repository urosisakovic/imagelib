import numpy as np


def imresize(img, height, width, inter='nn'):
    """
    Function which resizes image.

    Args:
        image
        height
        width
        inter
    Ret:
        Resized image.
    """
    assert inter in ['nn', 'bilinear', 'bicubic'], 'Invalid interpolation'

    # extract old image dimensions
    h, w = img.shape

    y_ratio = height / h
    x_ratio = width / w

    affine_A = np.array([[x_ratio, 0      ],
                         [0,       y_ratio]])
    affine_b = np.zeros((2, 1))
    affine_transform = np.concatenate([affine_A, affine_b], axis=1)

    return imaffine(img, affine_transform, inter)


def imrotate(img, angle, inter='nn', fill=0):
    """
    Rotates an image.

    Args:
        img:
        angle:
        inter:
        fill:
    Ret:
        Rotated image.
    """
    cos = np.cos(angle)
    sin = np.sin(angle)

    affine_A = np.array([[cos, -sin],
                         [sin,  cos]])
    affine_b = np.zeros((2, 1))
    affine_transform = np.concatenate([affine_A, affine_b], axis=1)

    return imaffine(img, affine_transform, inter, fill)


def imaffine(img, transformation, inter='nn', fill=0):
    """
    Applies given affine transformation to the image.

    Args:
        img:
        transformation:
        inter:
        fill:
    Rets:
        Result of the given affine transformation.
    """
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
        for (proj_coord, coord) in zip(projected_new_coords, image_coord_sys_new_img_coords):

            if proj_coord[0] < 0 or proj_coord[0] > img_h or\
                proj_coord[1] < 0 or proj_coord[1] > img_w:

                coord = coord.astype(np.int32)
                new_img[coord[0], coord[1]] = fill
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


# TODO
def fit_affine(a, b):
    """
    Given two sets of corresponding points, this function finds parameters
    of an affine transformation which minimizes MSE.
    """
    pass