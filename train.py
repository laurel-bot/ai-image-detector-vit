"""
train.py — Fine-tune ViT for AI-Generated Image Detection

Usage:
    python train.py --data_dir ./data --epochs 10 --batch_size 32 --lr 2e-4
"""

import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import create_dataloaders
from model import create_model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(input_tensor)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train ViT AI Image Detector")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root dir")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--save_path", type=str, default="ai_vit_detector.pth", help="Model save path")
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  AI Image Detector — ViT Training")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    # --- Data ---
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # --- Model ---
    model = create_model(num_classes=len(class_names), freeze_backbone=True)
    model = model.to(device)

    # --- Optimizer & Loss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training Loop ---
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✅ Best model saved → {args.save_path} (acc: {val_acc:.4f})")

        print()

    print(f"{'='*50}")
    print(f"  Training complete! Best Val Acc: {best_val_acc:.4f}")
    print(f"  Model saved to: {args.save_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
