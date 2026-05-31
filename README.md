# 🔍 AI Image Detector — Vision Transformer (ViT)

A lightweight, high-accuracy AI-generated image detector built on **Vision Transformer (ViT)**.

Fine-tunes `vit_base_patch16_224` (pretrained on ImageNet-21k) to perform binary classification:
- **Real** photographs
- **AI-generated** images (Stable Diffusion, DALL·E, Midjourney, etc.)

## 📁 Project Structure

```
ai-image-detector-vit/
├── model.py            # ViT model wrapper
├── dataset.py          # Data loading & augmentation
├── train.py            # Training / fine-tuning script
├── predict.py          # Single-image inference
├── requirements.txt    # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your dataset in **ImageFolder** format:

```
data/
├── train/
│   ├── real/        # Real photographs
│   └── ai/          # AI-generated images
└── val/
    ├── real/
    └── ai/
```

### 3. Train

```bash
python train.py --data_dir ./data --epochs 10 --batch_size 32 --lr 2e-4
```

The trained model will be saved as `ai_vit_detector.pth`.

### 4. Predict

```bash
python predict.py --image path/to/image.jpg
```

## 🏗️ Architecture

| Component | Detail |
|-----------|--------|
| Backbone | `vit_base_patch16_224` (timm) |
| Input | 224 × 224, center-cropped from 256 |
| Strategy | Freeze backbone, train classifier head |
| Classes | 2 (real / ai) |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |

## 📊 HD Image Handling

High-resolution images are gracefully downscaled:

```
HD Image (e.g. 4096×2160)
  → Resize shortest edge to 256
  → CenterCrop 224×224
  → Normalize (ImageNet stats)
  → Feed into ViT
```

## 📄 License

MIT License
