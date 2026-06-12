"""
Mars Rover Mission - Advanced Convolutional Neural Network (CNN) Vision Module
Upgraded to a 4-layer deep network for high-accuracy 8-class classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarsTerrainCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(MarsTerrainCNN, self).__init__()
        
        # Block 1: Input (3, 128, 128) -> Output (32, 64, 64) after pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: Input (32, 64, 64) -> Output (64, 32, 32) after pooling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: Input (64, 32, 32) -> Output (128, 16, 16) after pooling
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4: Input (128, 16, 16) -> Output (256, 8, 8) after pooling
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dense Classification Layers
        # 256 channels * 8x8 spatial map = 16,384 features
        self.fc1 = nn.Linear(256 * 8 * 8, 512) # Increased dense units to 512
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Feature Extraction Layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 8 * 8)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

LABELS = [
    "Bright dune",
    "Craters",
    "Dark dune",
    "Impact ejecta",
    "Other",
    "Slope streak",
    "Spider",
    "Swiss cheese"
]

def predict_terrain(model, image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        return LABELS[predicted_idx.item()]