import torch
from PIL import Image

from aesthetics_predictor import AestheticsPredictorV1
from transformers import CLIPProcessor, CLIPModel
import ImageReward as RM


device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# Load ImageReward (for human preference score)
# -------------------------------------------------
image_reward = RM.load("ImageReward-v1.0")


# -------------------------------------------------
# Load Aesthetic Predictor v0.1.2 (CLIP-based)
# This version uses ViT-B/32 under the hood.
# -------------------------------------------------
model_id = "shunk031/aesthetics-predictor-v1-vit-base-patch32"
aesthetic_model = AestheticsPredictorV1.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# -------------------------------------------------
# Compute ImageReward
# -------------------------------------------------
def compute_image_reward(image_path, prompt):
    scores = image_reward.score(prompt, [image_path])
    return float(scores)


# -------------------------------------------------
# Compute Aesthetic Score (simple-aesthetics-predictor 0.1.2)
# Note: Higher scores indicate better aesthetics (typically 0-10 range)
# -------------------------------------------------
def compute_aesthetic_score(image_path):    
    image = Image.open(image_path).convert("RGB")    
    # Preprocess for CLIP    
    inputs = processor(images=image, return_tensors="pt").to(device)    
    # Predict aesthetic score directly    
    with torch.no_grad():        
        outputs = aesthetic_model(**inputs)        
        score = outputs.logits.item() if hasattr(outputs, 'logits') else outputs.item()    
        
    return float(score)


# -------------------------------------------------
# Combined evaluation
# -------------------------------------------------
def evaluate_image(image_path, prompt):
    return {
        "image_reward": compute_image_reward(image_path, prompt),
        "aesthetic_score": compute_aesthetic_score(image_path),
    }


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    img = "esd_golden.png"
    prompt = "a painting of a runner"

    scores = evaluate_image(img, prompt)
    print(scores)
