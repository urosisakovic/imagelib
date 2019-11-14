import numpy as np

import utility


def main():
    image_path = '/home/uros/Desktop/img.jpg'
    scaled_down_img_path = '/home/uros/Desktop/scaled_down_img.jpg'
    img = utility.load_image(image_path, grayscale=True)

    utility.save_image(image_path, img)

    print('image shape: ', img.shape)

    h, w = img.shape

    new_img = [[[] for _ in range(w // 2)] for _ in range(h // 2)]

    for i in range(h):
        for j in range(w):
            new_img[i // 2][j // 2].append(img[i][j])

    for i in range(h // 2):
        for j in range(w // 2):
            new_img[i][j] = np.mean(new_img[i][j])

    new_img = np.array(new_img)

    print('scaled down image shape: ', new_img.shape)

    utility.save_image(scaled_down_img_path, new_img)

if __name__ == '__main__':
    main()