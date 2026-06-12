# Save this as train_vision.py in your main mars_rover_mission folder
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from src.vision import MarsTerrainCNN

def train_model():
    print("====================================================")
    print("🛸 DEEP MARS ROVER VISION TRAINING SYSTEM (CUDA) 🛸")
    print("====================================================\n")

    # 1. High-Performance Hyperparameters
    BATCH_SIZE = 32          
    EPOCHS = 20              
    LEARNING_RATE = 0.001
    DATA_DIR = os.path.join("data", "raw")

    # Environment Validation
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print("❌ Error: 'data/raw/' folder is empty or missing!")
        print("Please ensure your 8 dataset folders are placed inside data/raw/.")
        return

    # Hardware Optimization Allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Utilizing Training Hardware: [{str(device).upper()}]")

    # 2. Domain-Specific Data Transformations
    # Augmented transform for the training split to eliminate underfitting
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),   
        transforms.RandomRotation(15),     
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Pristine transform for the validation evaluation split
    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 3. Clean Dataset Splitting (Prevents Train-to-Validation Data Leakage)
    print("📦 Parsing planetary imagery and preparing subsets...")
    
    # Load separate instances to apply unique transforms to subsets correctly
    base_train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    base_val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transforms)
    
    total_images = len(base_train_dataset)
    train_size = int(0.8 * total_images)
    
    # Generate deterministic index permutations
    torch.manual_seed(42)
    indices = torch.randperm(total_images).tolist()
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    
    # Construct Subsets
    train_dataset = Subset(base_train_dataset, train_idx)
    val_dataset = Subset(base_val_dataset, val_idx)

    # Multi-threaded data loaders for GPU throughput optimization
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"-> Dataset Classes Recognized: {base_train_dataset.classes}")
    print(f"-> Total Image Assets: {total_images}")
    print(f"-> Training Partition Size: {len(train_dataset)}")
    print(f"-> Validation Partition Size: {len(val_dataset)}\n")

    # 4. Network and Optimization Tool Initialization
    num_detected_classes = len(base_train_dataset.classes)
    model = MarsTerrainCNN(num_classes=num_detected_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Gentle step decay to stabilize deep feature learning weights
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # 5. Core Deep Learning Training Loop
    print("🏋️ Commencing Advanced Training and Validation Runs...")
    print("--------------------------------------------------------------------------------")
    
    for epoch in range(1, EPOCHS + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass execution
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation execution
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tracking statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Learning Rate Adjuster step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_preds / total_samples) * 100
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, val_pred = torch.max(val_outputs, 1)
                val_correct += (val_pred == val_labels).sum().item()
                val_total += val_labels.size(0)
        
        val_acc = (val_correct / val_total) * 100
        
        # Display performance logs for the current epoch step
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | LR: {current_lr:.5f} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print("--------------------------------------------------------------------------------")
    print("✅ High-capacity optimization training cycle complete!")
    
    # 6. Secure Model Weights Storage
    output_model_dir = os.path.join("outputs", "models")
    os.makedirs(output_model_dir, exist_ok=True)
    output_model_path = os.path.join(output_model_dir, "mars_terrain_cnn.pth")
    
    torch.save(model.state_dict(), output_model_path)
    print(f"💾 Optimized structural weights file saved to: {output_model_path}")
    print("====================================================")

if __name__ == "__main__":
    train_model()