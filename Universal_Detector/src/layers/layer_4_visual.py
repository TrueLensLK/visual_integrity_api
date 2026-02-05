import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet  # <--- REQUIRED for your .pth file
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Point to YOUR specific file
MODEL_FILE = "efficientnet_b0.pth" 
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILE)

# Global variable to hold the loaded model
model = None

def load_custom_model():
    global model
    if model is not None: return model

    if not os.path.exists(MODEL_PATH):
        print(f"[Layer 4] Missing file: {MODEL_PATH}")
        return None

    try:
        print(f"[Layer 4] Loading your custom model: {MODEL_FILE}...")
        
        # Create the empty brain structure (EfficientNet-B0)
        # We must tell it how many classes you trained it on.
        # usually 2 (Fake vs Real) or 1 (Binary). Let's assume 2 for now.
        model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
        
        # B. Load YOUR memories (weights) into the brain
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        model.eval() # Set to "Test Mode"
        print("[Layer 4] Custom Model Loaded Successfully.")
        return model

    except Exception as e:
        print(f"[Layer 4 Error] Could not load .pth file: {e}")
        return None

def predict_visuals(file_path):
    """
    Layer 4: Uses YOUR Custom EfficientNet .pth file
    """
    net = load_custom_model()
    if net is None: return 0

    try:
        # 1. Preprocess (Must match how you trained it!)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(file_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0) # Add batch dimension

        # 2. Predict
        with torch.no_grad():
            outputs = net(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # CHECK YOUR TRAINING LABELS! 
            # Usually: 0=Fake, 1=Real. If your results are backward, swap these numbers.
            prob_fake = probs[0][0].item()
            prob_real = probs[0][1].item()

            print(f"   [My Model] Fake: {prob_fake:.1%} | Real: {prob_real:.1%}")

            # 3. Score
            if prob_fake > 0.90: return -50
            if prob_fake > 0.70: return -35
            if prob_fake > 0.55: return -20
            
            if prob_real > 0.90: return 30
            if prob_real > 0.75: return 15
            
            return 0

    except Exception as e:
        print(f"   [Layer 4 Error] {e}")
        return 0