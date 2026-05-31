"""
predict.py — Single-Image Inference for AI Image Detection

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image photo.png --model ai_vit_detector.pth
"""

import argparse

import torch
from PIL import Image

from dataset import get_transforms
from model import create_model

# Class labels — must match training order (ImageFolder sorts alphabetically)
CLASS_NAMES = ["ai", "real"]


def predict(image_path: str, model_path: str = "ai_vit_detector.pth") -> dict:
    """Run inference on a single image.

    Args:
        image_path: Path to the input image (supports HD images).
        model_path: Path to the trained model weights.

    Returns:
        dict with keys: predicted_class, confidence, probabilities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    model = create_model(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # --- Preprocess Image ---
    transform = get_transforms(is_training=False)
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # --- Inference ---
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()  # [num_classes]

    predicted_idx = probs.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probs[predicted_idx].item()

    return {
        "image": image_path,
        "original_size": f"{original_size[0]}x{original_size[1]}",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {name: probs[i].item() for i, name in enumerate(CLASS_NAMES)},
    }


def main():
    parser = argparse.ArgumentParser(description="AI Image Detector — Predict")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="ai_vit_detector.pth", help="Model weights path")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  AI Image Detector — ViT Inference")
    print(f"{'='*50}\n")

    result = predict(args.image, args.model)

    label_emoji = "🤖" if result["predicted_class"] == "ai" else "📷"
    print(f"  Image:       {result['image']}")
    print(f"  Resolution:  {result['original_size']}")
    print(f"  Prediction:  {label_emoji} {result['predicted_class'].upper()}")
    print(f"  Confidence:  {result['confidence']:.2%}")
    print(f"  ─────────────────────────")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:>5}: {prob:.2%} {bar}")
    print()


if __name__ == "__main__":
    main()
