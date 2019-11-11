import utility

image_path = '/home/uros/Desktop/cat.jpg'
    
rgb_image = utility.load_image(image_path)

grayscale_image = utility.rgb_to_grayscale(rgb_image)

print(grayscale_image.shape)