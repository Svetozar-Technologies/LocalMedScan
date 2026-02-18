#!/usr/bin/env python3
"""Train MobileNetV2 for malaria parasite detection on NIH dataset.

Usage:
    python scripts/train_malaria_model.py --data-dir test_data/malaria/cell_images --epochs 3

This fine-tunes a pretrained ImageNet MobileNetV2 for binary classification
(Parasitized vs Uninfected) on the NIH Malaria Cell Images Dataset.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def main():
    parser = argparse.ArgumentParser(description="Train malaria detection model.")
    parser.add_argument("--data-dir", required=True, help="Path to cell_images/ directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--output", default=None, help="Output path for model weights")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    full_dataset = datasets.ImageFolder(args.data_dir)

    # Class mapping
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to idx: {full_dataset.class_to_idx}")
    print(f"Total images: {len(full_dataset)}")

    # Split: 80% train, 10% val, 10% test
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    print("\nLoading pretrained MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze early layers, fine-tune later layers
    for param in model.features[:14].parameters():
        param.requires_grad = False

    # Replace classifier head for binary classification
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {running_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*correct/total:.1f}%")

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s)")
        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        scheduler.step()

    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    test_correct = 0
    test_total = 0
    tp = fp = tn = fn = 0

    # class_to_idx: Parasitized=0, Uninfected=1
    parasitized_idx = full_dataset.class_to_idx.get("Parasitized", 0)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            for pred, true in zip(predicted, labels):
                if true == parasitized_idx and pred == parasitized_idx:
                    tp += 1
                elif true != parasitized_idx and pred == parasitized_idx:
                    fp += 1
                elif true != parasitized_idx and pred != parasitized_idx:
                    tn += 1
                else:
                    fn += 1

    test_acc = 100. * test_correct / test_total
    sensitivity = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = 100. * tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\n  Test Accuracy:  {test_acc:.1f}%")
    print(f"  Sensitivity:    {sensitivity:.1f}%")
    print(f"  Specificity:    {specificity:.1f}%")
    print(f"  Precision:      {precision:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"                   Predicted +  Predicted -")
    print(f"  Actual +         {tp:>10}  {fn:>10}")
    print(f"  Actual -         {fp:>10}  {tn:>10}")

    # Save model
    from core.model_manager import get_model_manager
    mm = get_model_manager()

    if args.output:
        output_path = Path(args.output)
    else:
        model_dir = mm.ensure_model_dir("malaria-mobilenetv2")
        output_path = model_dir / "model.pth"

    # Save only the model state dict
    torch.save(best_state, str(output_path))
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Model saved to: {output_path}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Best validation accuracy: {best_val_acc:.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
