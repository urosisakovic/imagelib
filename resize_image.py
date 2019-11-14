import numpy as np
import utility

# TODO: Enable bilinear and bicubic interpolations.
# TODO: Inspect why is it so slow. Inspect how libraries manage this with big resolution images.

def imresize(img, height, width, interpolation='nn'):
    assert interpolation in ['nn', 'bilinear', 'bicubic'], 'Invalid interpolation'

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

    return resized_image
    
    if interpolation == 'nn':
        pass

    elif interpolation == 'bilinear':
        pass

    elif interpolation == 'bicubic':
        pass

    return resized_image_indices

if __name__ == '__main__':
    img = utility.load_image('/home/uros/Desktop/small_img.jpg', grayscale=True)

    h, w = 100, 10

    resized_img = imresize(img, h, w)

    utility.save_image('/home/uros/Desktop/nn_resized_img_{}x{}.jpg'.format(h, w), resized_img)