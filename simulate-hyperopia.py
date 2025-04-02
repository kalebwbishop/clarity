import cv2
import numpy as np
import os
import glob

def simulate_hyperopia(image_path, output_path, blur_strength=15):
    # Read the image
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a mask to apply a blur to the center of the image
    mask = np.zeros((height, width), dtype=np.float32)

    # Define the blur region (closer to the center of the image)
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
            mask[y, x] = np.exp(-distance / (width / 4))  # Use exponential decay for blur strength

    # Normalize the mask to get values between 0 and 1
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)

    # Apply the blur to the image based on the mask
    blurred_image = np.zeros_like(image)
    for c in range(3):  # Loop through each color channel
        blurred_image[..., c] = cv2.GaussianBlur(image[..., c], (blur_strength, blur_strength), 0)
    
    # Convert image to float32 for proper blending
    image_float = image.astype(np.float32)
    blurred_image_float = blurred_image.astype(np.float32)
    
    # Blend the blurred image with the original based on the mask
    # Create a 3-channel mask for RGB
    mask_3d = np.stack([mask] * 3, axis=-1)
    
    # Perform the blending manually
    final_image = image_float * (1 - mask_3d) + blurred_image_float * mask_3d
    
    # Convert back to uint8 for saving
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    # Save the output image
    cv2.imwrite(output_path, final_image)

    # Optionally, display the image (if you want to see the result immediately)
    # cv2.imshow('Hyperopia Simulation', final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('dataset/hyperopia', exist_ok=True)

    # Get all input images
    input_images = glob.glob('dataset/inputs/input_*.png')
    
    # Process each image
    for input_path in input_images:
        # Extract the number from the filename
        filename = os.path.basename(input_path)
        number = filename.split('_')[1].split('.')[0]
        
        # Create output path
        output_path = f'dataset/hyperopia/output_{number}.png'
        
        # Process the image
        simulate_hyperopia(input_path, output_path)
        print(f"Processed {input_path} -> {output_path}")
