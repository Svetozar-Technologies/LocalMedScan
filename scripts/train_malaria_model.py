#!/usr/bin/env python3
"""Train MobileNetV2 for malaria parasite detection — optimized for maximum accuracy.

Usage:
    python scripts/train_malaria_model.py --data-dir test_data/malaria/cell_images
    python scripts/train_malaria_model.py --data-dir test_data/malaria/cell_images --epochs 30 --batch-size 64

Key improvements over baseline:
    - Fixed dataset transform bug (train/val/test now get independent transforms)
    - CUDA auto-detection (uses GPU when available)
    - 3-phase progressive layer unfreezing
    - Strong data augmentation (RandomResizedCrop, RandomAffine, GaussianBlur, RandomErasing)
    - Cosine annealing with warm restarts
    - Mixed precision training (AMP) for speed on GPU
    - Label smoothing (0.1)
    - Gradient clipping (max_norm=1.0)
    - Test-time augmentation (TTA) for final evaluation
    - Early stopping with patience
    - TensorBoard logging (loss, accuracy, LR, confusion matrix, sample predictions)

Monitor training:
    tensorboard --logdir runs/
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid


# ---------------------------------------------------------------------------
# TransformSubset — fixes the critical bug where train/val/test shared
# the same parent dataset and overwrote each other's transforms.
# ---------------------------------------------------------------------------
class TransformSubset(Dataset):
    """Wraps a Subset with its own independent transform."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------
def tta_predict(model, images, device):
    """Run test-time augmentation: original + 3 flips, average predictions."""
    with torch.inference_mode():
        # Original
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        # Horizontal flip
        logits_hflip = model(torch.flip(images, dims=[3]))
        probs += torch.softmax(logits_hflip, dim=1)

        # Vertical flip
        logits_vflip = model(torch.flip(images, dims=[2]))
        probs += torch.softmax(logits_vflip, dim=1)

        # Both flips
        logits_hvflip = model(torch.flip(images, dims=[2, 3]))
        probs += torch.softmax(logits_hvflip, dim=1)

        probs /= 4.0
    return probs


# ---------------------------------------------------------------------------
# Freeze/Unfreeze helpers
# ---------------------------------------------------------------------------
def freeze_backbone(model):
    """Freeze all backbone parameters, only train classifier."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_top_layers(model):
    """Unfreeze features[14:] (last 5 inverted residual blocks)."""
    for param in model.features[:14].parameters():
        param.requires_grad = False
    for param in model.features[14:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    max_grad_norm=1.0, writer=None, global_step_offset=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # TensorBoard: per-batch loss
        if writer is not None:
            global_step = global_step_offset + batch_idx
            writer.add_scalar("batch/train_loss", loss.item(), global_step)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f} "
                  f"Acc: {100.*correct/total:.1f}%")

    return running_loss / len(loader), 100. * correct / total


@torch.inference_mode()
def evaluate(model, loader, criterion, device, use_tta=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_tta:
            probs = tta_predict(model, images, device)
            _, predicted = probs.max(1)
            # compute loss on original only for consistency
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)

        running_loss += loss.item()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train malaria detection model (max accuracy).")
    parser.add_argument("--data-dir", required=True, help="Path to cell_images/ directory")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate for phase 1")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (default: 7)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--output", default=None, help="Output path for model weights")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--logdir", default="runs", help="TensorBoard log directory (default: runs/)")
    args = parser.parse_args()

    # --- TensorBoard ---
    run_name = f"malaria_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = Path(args.data_dir).parent.parent.parent / args.logdir / run_name
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"TensorBoard logs: {log_path}")
    print(f"  Monitor with: tensorboard --logdir {log_path.parent}")

    # Log hyperparameters
    writer.add_text("hyperparameters", f"""
| Parameter | Value |
|-----------|-------|
| epochs | {args.epochs} |
| batch_size | {args.batch_size} |
| lr | {args.lr} |
| patience | {args.patience} |
| workers | {args.workers} |
| tta | {not args.no_tta} |
""")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Dataset ---
    print(f"\nLoading dataset from {args.data_dir}...")

    # Load without transform (we apply transforms via TransformSubset)
    full_dataset = datasets.ImageFolder(args.data_dir)
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to idx: {full_dataset.class_to_idx}")
    print(f"Total images: {len(full_dataset)}")

    # Split: 80% train, 10% val, 10% test
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap with independent transforms (fixes the critical bug)
    train_dataset = TransformSubset(train_subset, get_train_transform())
    val_dataset = TransformSubset(val_subset, get_val_transform())
    test_dataset = TransformSubset(test_subset, get_val_transform())

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
    )

    # --- Model ---
    print("\nLoading pretrained MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # ===================================================================
    # PHASE 1: Train classifier head only (epochs 1–5)
    # ===================================================================
    phase1_epochs = min(5, args.epochs)
    print(f"\n{'='*60}")
    print(f"PHASE 1: Classifier head only ({phase1_epochs} epochs, LR={args.lr})")
    print(f"{'='*60}")

    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=phase1_epochs, T_mult=1)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    global_epoch = 0  # continuous epoch counter across all phases

    for epoch in range(1, phase1_epochs + 1):
        global_epoch += 1
        epoch_start = time.time()
        batch_offset = (global_epoch - 1) * len(train_loader)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            writer=writer, global_step_offset=batch_offset,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{phase1_epochs} ({elapsed:.0f}s) LR={current_lr:.2e}")
        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")

        # TensorBoard: epoch-level metrics
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, global_epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, global_epoch)
        writer.add_scalar("learning_rate", current_lr, global_epoch)
        writer.add_scalar("phase", 1, global_epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

    # ===================================================================
    # PHASE 2: Unfreeze top layers (epochs 6–15)
    # ===================================================================
    phase2_epochs = min(10, max(0, args.epochs - phase1_epochs))
    if phase2_epochs > 0:
        phase2_lr = 3e-4
        print(f"\n{'='*60}")
        print(f"PHASE 2: Unfreeze features[14:] ({phase2_epochs} epochs, LR={phase2_lr})")
        print(f"{'='*60}")

        # Load best state from phase 1
        if best_state:
            model.load_state_dict(best_state)
            model = model.to(device)

        unfreeze_top_layers(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=phase2_lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=phase2_epochs, T_mult=1)
        no_improve = 0

        for epoch in range(1, phase2_epochs + 1):
            global_epoch += 1
            epoch_start = time.time()
            batch_offset = (global_epoch - 1) * len(train_loader)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch,
                writer=writer, global_step_offset=batch_offset,
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            elapsed = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{phase2_epochs} ({elapsed:.0f}s) LR={current_lr:.2e}")
            print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
            print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")

            # TensorBoard: epoch-level metrics
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, global_epoch)
            writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, global_epoch)
            writer.add_scalar("learning_rate", current_lr, global_epoch)
            writer.add_scalar("phase", 2, global_epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

    # ===================================================================
    # PHASE 3: Unfreeze all layers (epochs 16–30) with early stopping
    # ===================================================================
    phase3_epochs = max(0, args.epochs - phase1_epochs - phase2_epochs)
    if phase3_epochs > 0:
        phase3_lr = 1e-5
        print(f"\n{'='*60}")
        print(f"PHASE 3: Full fine-tune ({phase3_epochs} epochs, LR={phase3_lr}, patience={args.patience})")
        print(f"{'='*60}")

        # Load best state from phase 2
        if best_state:
            model.load_state_dict(best_state)
            model = model.to(device)

        unfreeze_all(model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=phase3_lr, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=phase3_epochs, T_mult=1)
        no_improve = 0

        for epoch in range(1, phase3_epochs + 1):
            global_epoch += 1
            epoch_start = time.time()
            batch_offset = (global_epoch - 1) * len(train_loader)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch,
                writer=writer, global_step_offset=batch_offset,
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            elapsed = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{phase3_epochs} ({elapsed:.0f}s) LR={current_lr:.2e}")
            print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
            print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")

            # TensorBoard: epoch-level metrics
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, global_epoch)
            writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, global_epoch)
            writer.add_scalar("learning_rate", current_lr, global_epoch)
            writer.add_scalar("phase", 3, global_epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break

    # ===================================================================
    # TEST EVALUATION
    # ===================================================================
    print(f"\n{'='*60}")
    print("TEST EVALUATION")
    print(f"{'='*60}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    model.eval()

    use_tta = not args.no_tta
    print(f"Test-time augmentation: {'ON (4 views)' if use_tta else 'OFF'}")

    # Detailed evaluation with confusion matrix
    test_correct = 0
    test_total = 0
    tp = fp = tn = fn = 0

    parasitized_idx = full_dataset.class_to_idx.get("Parasitized", 0)

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if use_tta:
                probs = tta_predict(model, images, device)
                _, predicted = probs.max(1)
            else:
                with autocast(device_type="cuda", enabled=(device.type == "cuda")):
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
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"\n  Test Accuracy:  {test_acc:.2f}% ({test_correct}/{test_total})")
    print(f"  Sensitivity:    {sensitivity:.2f}%")
    print(f"  Specificity:    {specificity:.2f}%")
    print(f"  Precision:      {precision:.2f}%")
    print(f"  F1 Score:       {f1:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"                   Predicted +  Predicted -")
    print(f"  Actual +         {tp:>10}  {fn:>10}")
    print(f"  Actual -         {fp:>10}  {tn:>10}")

    # Also run without TTA to show difference
    if use_tta:
        _, test_acc_no_tta = evaluate(model, test_loader, criterion, device, use_tta=False)
        print(f"\n  (Without TTA:   {test_acc_no_tta:.2f}%)")

    # TensorBoard: final test metrics + confusion matrix
    writer.add_scalar("test/accuracy", test_acc, global_epoch)
    writer.add_scalar("test/sensitivity", sensitivity, global_epoch)
    writer.add_scalar("test/specificity", specificity, global_epoch)
    writer.add_scalar("test/precision", precision, global_epoch)
    writer.add_scalar("test/f1", f1, global_epoch)

    # Log confusion matrix as a figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        cm = np.array([[tp, fn], [fp, tn]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted +", "Predicted -"])
        ax.set_yticklabels(["Actual +", "Actual -"])
        ax.set_title(f"Test Confusion Matrix (Acc: {test_acc:.2f}%)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
        fig.colorbar(im)
        fig.tight_layout()
        writer.add_figure("test/confusion_matrix", fig, global_epoch)
        plt.close(fig)
    except ImportError:
        pass  # matplotlib not installed, skip figure

    # Log sample predictions from test set
    try:
        sample_images, sample_labels = next(iter(test_loader))
        sample_images = sample_images.to(device)
        with torch.inference_mode():
            sample_probs = torch.softmax(model(sample_images), dim=1)
            _, sample_preds = sample_probs.max(1)

        # Denormalize for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        display_imgs = sample_images[:16].cpu() * std + mean
        display_imgs = display_imgs.clamp(0, 1)

        classes = full_dataset.classes
        img_labels = [
            f"{'OK' if p == t else 'WRONG'} pred={classes[p]} true={classes[t]}"
            for p, t in zip(sample_preds[:16].cpu(), sample_labels[:16])
        ]

        grid = make_grid(display_imgs, nrow=4, normalize=False)
        writer.add_image("test/sample_predictions", grid, global_epoch)
        writer.add_text("test/sample_labels", "\n".join(img_labels), global_epoch)
    except Exception:
        pass  # non-critical

    # Log hparams with final metrics
    writer.add_hparams(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "tta": not args.no_tta,
        },
        {
            "hparam/test_accuracy": test_acc,
            "hparam/best_val_accuracy": best_val_acc,
            "hparam/test_f1": f1,
            "hparam/test_sensitivity": sensitivity,
            "hparam/test_specificity": specificity,
        },
    )

    # --- Save model ---
    from core.model_manager import get_model_manager
    mm = get_model_manager()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        model_dir = mm.ensure_model_dir("malaria-mobilenetv2")
        output_path = model_dir / "model.pth"

    torch.save(best_state, str(output_path))
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Model saved to: {output_path}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")

    writer.close()
    print(f"\n  TensorBoard logs at: {log_path}")
    print(f"  Run: tensorboard --logdir {log_path.parent}")

    print("\nDone!")


if __name__ == "__main__":
    main()
