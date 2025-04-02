import numpy as np
from PIL import Image

def generate_images(image_size=100, num_images=10, output_to_file=False):
    for i in range(num_images):
        # Create random grayscale image
        features = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)

        # Create and save the raw image if needed
        image = Image.fromarray(features)
        if output_to_file:
            image.save(f'dataset/inputs/input_{i}.png')


if __name__ == "__main__":
    generate_images(100, 10000, True)

