import utility

image_path = '/home/uros/Desktop/cat.jpg'
grayscale_path = '/home/uros/Desktop/grayscale_cat.jpg'
    
rgb_image = utility.load_image(image_path)

grayscale_image = utility.rgb_to_grayscale(rgb_image)

print(grayscale_image.shape)

utility.save_image(grayscale_path, grayscale_image)

img = utility.load_image(grayscale_path)

print(img.shape)