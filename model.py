"""
model.py — ViT Model Wrapper

Loads vit_base_patch16_224 via timm, freezes the backbone,
and replaces the classification head for binary detection.
"""

import timm
import torch.nn as nn


def create_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """Create a ViT model for AI image detection.

    Args:
        num_classes: Number of output classes (default: 2 → real/ai).
        freeze_backbone: If True, freeze all layers except the classifier head.

    Returns:
        A ready-to-train nn.Module.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] vit_base_patch16_224 loaded")
    print(f"[Model] Trainable params: {trainable:,} / {total:,}")

    return model
