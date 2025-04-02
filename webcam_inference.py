import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from modelHyperopia import HyperopiaModel
import time

def load_model(model_path, device):
    """Load the trained model from disk"""
    model = HyperopiaModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def preprocess_frame(frame):
    """Preprocess a single frame from webcam"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img = Image.fromarray(frame_rgb)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = transform(img)
    
    # Add batch dimension
    img = img.unsqueeze(0)
    
    return img

def postprocess_output(tensor):
    """Convert model output tensor to image"""
    # Remove batch dimension and convert to numpy
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Convert from [0,1] to [0,255]
    img = (img * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def find_available_camera():
    """Try to find an available camera"""
    print("Searching for available cameras...")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Try DirectShow backend
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Found working camera at index {i}")
                cap.release()
                return i
            cap.release()
        time.sleep(0.1)  # Small delay to prevent overwhelming the system
    
    # Try without backend specification
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Found working camera at index {i}")
                cap.release()
                return i
            cap.release()
        time.sleep(0.1)
    
    return None

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'best_model_hyperopia.pth'
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find available camera
    camera_index = find_available_camera()
    if camera_index is None:
        print("No working camera found. Please check your camera connection and permissions.")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Preprocess frame
            input_tensor = preprocess_frame(frame)
            input_tensor = input_tensor.to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
            
            # Convert output to image
            output_img = postprocess_output(output)
            
            # Resize output to match input frame size
            output_img = cv2.resize(output_img, (frame.shape[1], frame.shape[0]))
            
            # Display the original and corrected frames side by side
            combined = np.hstack((frame, output_img))
            cv2.imshow('Original (Left) vs Corrected (Right)', combined)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 