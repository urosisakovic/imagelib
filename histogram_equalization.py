import matplotlib.pyplot as plt

import utility

image_path = '/home/uros/Desktop/mnt.png'

img = utility.load_image(image_path, grayscale=True)

img *= 255

plt.hist(img.flatten(), 255)

utility.save_plot(plt, '/home/uros/Desktop')
