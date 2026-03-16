"""
Custom AI Image Detector Training Script
=========================================
Fine-tune EfficientNet-B0 on your own AI vs Real dataset.

Usage:
    1. Organize images in datasets/train/{real,ai}/ and datasets/val/{real,ai}/
    2. Run: python train_custom_model.py
    3. Model saved to: Universal_Detector/src/layers/custom_detector.pth

Requirements:
    pip install torch torchvision timm pillow tqdm tensorboard
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Dataset paths
    "train_dir": "datasets/train",
    "val_dir": "datasets/val",
    
    # Model settings
    "model_name": "efficientnet_b0",  # or "vit_base_patch16_224", "convnext_tiny"
    "num_classes": 2,  # real=0, ai=1
    "pretrained": True,
    
    # Training settings
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    
    # Image settings
    "image_size": 224,
    
    # Output
    "output_path": "Universal_Detector/src/layers/custom_detector.pth",
    "checkpoint_dir": "checkpoints",
}


# ============================================================================
# DATASET CLASS
# ============================================================================
class AIvsRealDataset(Dataset):
    """Dataset for AI vs Real image classification."""
    
    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        self.samples = []
        self.class_to_idx = {"real": 0, "ai": 1}
        
        # Load all images
        for class_name in ["real", "ai"]:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found!")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        
        # Count per class
        real_count = sum(1 for _, label in self.samples if label == 0)
        ai_count = sum(1 for _, label in self.samples if label == 1)
        print(f"  Real: {real_count}, AI: {ai_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            image = Image.new("RGB", (CONFIG["image_size"], CONFIG["image_size"]))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# DATA AUGMENTATION
# ============================================================================
def get_transforms(is_train: bool):
    """Get transforms for training or validation."""
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(CONFIG["image_size"]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            # JPEG compression augmentation (simulates WhatsApp/social media)
            transforms.RandomApply([
                transforms.Lambda(lambda x: jpeg_compress(x, quality=random.randint(50, 95)))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


import random
from io import BytesIO

def jpeg_compress(image: Image.Image, quality: int = 75) -> Image.Image:
    """Simulate JPEG compression (data augmentation)."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ============================================================================
# MODEL CREATION
# ============================================================================
def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Create model using timm library."""
    
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    
    print(f"Created {model_name} with {num_classes} classes")
    return model


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class stats
            for i, label in enumerate(labels):
                class_total[label.item()] += 1
                if predicted[i] == label:
                    class_correct[label.item()] += 1
    
    overall_acc = 100. * correct / total
    real_acc = 100. * class_correct[0] / max(1, class_total[0])
    ai_acc = 100. * class_correct[1] / max(1, class_total[1])
    
    print(f"\n  Overall: {overall_acc:.2f}% | Real: {real_acc:.2f}% | AI: {ai_acc:.2f}%")
    
    return running_loss / len(dataloader), overall_acc


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train():
    """Main training function."""
    
    print("=" * 60)
    print("CUSTOM AI DETECTOR TRAINING")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if CUDA available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = AIvsRealDataset(
        CONFIG["train_dir"],
        transform=get_transforms(is_train=True),
        is_train=True
    )
    val_dataset = AIvsRealDataset(
        CONFIG["val_dir"],
        transform=get_transforms(is_train=False),
        is_train=False
    )
    
    if len(train_dataset) == 0:
        print("\nERROR: No training images found!")
        print("Please organize your images:")
        print("  datasets/train/real/  <- Put real photos here")
        print("  datasets/train/ai/    <- Put AI images here")
        print("  datasets/val/real/    <- Validation real")
        print("  datasets/val/ai/      <- Validation AI")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        CONFIG["model_name"],
        CONFIG["num_classes"],
        CONFIG["pretrained"]
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
        eta_min=1e-6
    )
    
    # Create checkpoint directory
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    # Training loop
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(CONFIG["epochs"]):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            
            # Save model weights only (smaller file)
            torch.save(model.state_dict(), CONFIG["output_path"])
            print(f"  [+] New best model saved! ({val_acc:.2f}%)")
            
            # Also save full checkpoint for resume
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": CONFIG,
            }
            torch.save(checkpoint, os.path.join(CONFIG["checkpoint_dir"], "best_checkpoint.pth"))
    
    # Save training history
    with open(os.path.join(CONFIG["checkpoint_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {CONFIG['output_path']}")
    print("\nTo use in the detector, the model will be automatically loaded")
    print("if 'custom_detector.pth' exists in the layers directory.")


if __name__ == "__main__":
    train()
