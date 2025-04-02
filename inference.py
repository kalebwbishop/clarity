import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
from modelHyperopia import HyperopiaModel

def load_model(model_path, device):
    """Load the trained model from disk"""
    model = HyperopiaModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image(image_path, transform):
    """Load and preprocess a single image"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img = Image.fromarray(img)
    
    # Apply transforms
    img = transform(img)
    
    # Add batch dimension
    img = img.unsqueeze(0)
    
    return img

def postprocess_image(tensor):
    """Convert model output tensor to image"""
    # Remove batch dimension and convert to numpy
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Convert from [0,1] to [0,255]
    img = (img * 255).astype(np.uint8)
    
    return img

def process_directory(input_dir, output_dir, model, transform, device):
    """Process all images in a directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    print(f"Processing {len(image_files)} images...")
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"corrected_{filename}")
        
        # Load and preprocess image
        img_tensor = preprocess_image(input_path, transform)
        img_tensor = img_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert output to image
        output_img = postprocess_image(output)
        
        # Save the result
        cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        print(f"Processed: {filename}")
        break

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms (must match training transforms)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Load model
    model_path = 'model_hyperopia.pth'
    model = load_model(model_path, device)
    
    # Process images
    input_dir = 'dataset/hyperopia'  # Directory containing images to process
    output_dir = 'dataset/corrected_hyperopia'  # Directory to save processed images
    
    process_directory(input_dir, output_dir, model, transform, device)
    print("Processing complete!")

if __name__ == "__main__":
    main() 