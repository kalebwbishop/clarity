import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class HyperopiaDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None, max_images=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform

        if max_images is None:
            self.files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
        else:
            self.files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])[:max_images]
        
        # Load all images into memory
        print(f"Loading first {max_images} images into memory...")
        self.input_images = []
        self.output_images = []
        
        for file in tqdm(self.files, desc="Loading images"):
            input_path = os.path.join(self.input_dir, file)
            output_path = os.path.join(self.output_dir, f"output_{file.split('_')[1]}")
            
            # Read images
            input_img = cv2.imread(input_path)
            output_img = cv2.imread(output_path)
            
            # Convert BGR to RGB
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            input_img = Image.fromarray(input_img)
            output_img = Image.fromarray(output_img)
            
            if self.transform:
                input_img = self.transform(input_img)
                output_img = self.transform(output_img)
            
            self.input_images.append(input_img)
            self.output_images.append(output_img)
        
        print(f"Loaded {len(self.input_images)} image pairs into memory")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.input_images[idx], self.output_images[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class HyperopiaModel(nn.Module):
    def __init__(self):
        super(HyperopiaModel, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # Upsampling blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Downsampling
        x = self.down1(x)
        x = self.down2(x)
        
        # Residual blocks
        x = self.res_blocks(x)
        
        # Upsampling
        x = self.up1(x)
        x = self.up2(x)
        
        # Final convolution
        x = self.final(x)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.train()
    best_val_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                running_loss += loss.item()
            
            train_loss = running_loss/len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_pixels = 0
            total_pixels = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    # Consider a pixel correct if it's within 5 of the target value
                    correct = 127 * torch.abs(outputs - targets) < 5
                    correct_pixels += correct.sum().item()
                    total_pixels += targets.numel()
            
            val_loss = val_loss/len(val_loader)
            val_accuracy = (correct_pixels / total_pixels) * 100
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model_hyperopia.pth')
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")
        torch.save(model.state_dict(), 'interrupted_model_hyperopia.pth')
        print("Model saved as 'interrupted_model_hyperopia.pth'")
        raise  # Re-raise the exception to exit the program

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and split into train and validation
    full_dataset = HyperopiaDataset(
        input_dir='dataset/inputs',
        output_dir='dataset/hyperopia',
        transform=transform,
        max_images=None
    )
    
    # Split dataset into train and validation (80-20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = HyperopiaModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device=device)
