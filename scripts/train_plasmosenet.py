#!/usr/bin/env python3
"""Train PlasmoSENet from scratch for malaria parasite detection.

Usage:
    python scripts/train_plasmosenet.py --data-dir test_data/malaria/cell_images
    python scripts/train_plasmosenet.py --data-dir test_data/malaria/cell_images --epochs 150 --batch-size 64

Custom architecture trained from scratch with:
    - AdamW optimizer with linear warmup + cosine annealing
    - Mixup/CutMix batch-level augmentation
    - Strong data augmentation (rotation 90, color jitter, erasing)
    - Stochastic depth, dropout, label smoothing, weight decay
    - 5-view test-time augmentation
    - TensorBoard logging

Monitor training:
    tensorboard --logdir runs/
"""

import argparse
import math
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
from torchvision import datasets, transforms
from torchvision.utils import make_grid


# ---------------------------------------------------------------------------
# TransformSubset
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
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Mixup / CutMix
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """Mixup: blend two random samples."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    _, _, H, W = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = max(0, cy - cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    return y1, x1, y2, x2


def cutmix_data(x, y, alpha=1.0):
    """CutMix: paste a patch from one sample onto another."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    y1, x1, y2, x2 = rand_bbox(x.size(), lam)
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(-2) * x.size(-1)))
    return x_mixed, y, y[index], lam


# ---------------------------------------------------------------------------
# TTA (5-view)
# ---------------------------------------------------------------------------
def tta_predict(model, images, device):
    """5-view TTA: original + 3 flips + 90 rotation."""
    with torch.inference_mode():
        probs = torch.softmax(model(images), dim=1)
        probs += torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        probs += torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
        probs += torch.softmax(model(torch.flip(images, dims=[2, 3])), dim=1)
        probs += torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)
        probs /= 5.0
    return probs


# ---------------------------------------------------------------------------
# LR Schedule: linear warmup + cosine annealing
# ---------------------------------------------------------------------------
def get_lr_lambda(warmup_epochs, total_epochs, min_lr, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    use_mixup=True, writer=None, global_step_offset=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Apply Mixup or CutMix
        if use_mixup:
            if np.random.random() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
        else:
            targets_a = targets_b = labels
            lam = 1.0

        optimizer.zero_grad()

        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if writer is not None:
            writer.add_scalar("batch/train_loss", loss.item(), global_step_offset + batch_idx)

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
            with autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            with autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)

        running_loss += loss.item()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train PlasmoSENet from scratch.")
    parser.add_argument("--data-dir", required=True, help="Path to cell_images/ directory")
    parser.add_argument("--epochs", type=int, default=150, help="Total training epochs (default: 150)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (default: 1e-3)")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR (default: 1e-6)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay (default: 0.05)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--output", default=None, help="Output path for model weights")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--no-mixup", action="store_true", help="Disable Mixup/CutMix")
    parser.add_argument("--logdir", default="runs", help="TensorBoard log directory (default: runs/)")
    parser.add_argument("--drop-path", type=float, default=0.2, help="Drop path rate (default: 0.2)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)")
    args = parser.parse_args()

    # --- TensorBoard ---
    run_name = f"plasmosenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = Path(args.data_dir).parent.parent.parent / args.logdir / run_name
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"TensorBoard logs: {log_path}")
    print(f"  Monitor with: tensorboard --logdir {log_path.parent}")

    writer.add_text("hyperparameters", f"""
| Parameter | Value |
|-----------|-------|
| architecture | PlasmoSENet (from scratch) |
| epochs | {args.epochs} |
| warmup | {args.warmup} |
| batch_size | {args.batch_size} |
| lr | {args.lr} |
| min_lr | {args.min_lr} |
| weight_decay | {args.weight_decay} |
| patience | {args.patience} |
| drop_path | {args.drop_path} |
| dropout | {args.dropout} |
| mixup | {not args.no_mixup} |
| tta | {not args.no_tta} |
""")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Dataset ---
    print(f"\nLoading dataset from {args.data_dir}...")
    full_dataset = datasets.ImageFolder(args.data_dir)
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to idx: {full_dataset.class_to_idx}")
    print(f"Total images: {len(full_dataset)}")

    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

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
    from core.plasmosenet import PlasmoSENet

    print(f"\nBuilding PlasmoSENet (drop_path={args.drop_path}, dropout={args.dropout})...")
    model = PlasmoSENet(num_classes=2, drop_path_rate=args.drop_path, dropout=args.dropout)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        get_lr_lambda(args.warmup, args.epochs, args.min_lr, args.lr),
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # --- Training ---
    print(f"\nTraining for {args.epochs} epochs (warmup: {args.warmup}, patience: {args.patience})...")
    print(f"Mixup/CutMix: {'ON' if not args.no_mixup else 'OFF'}")
    print(f"{'='*60}")

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        batch_offset = (epoch - 1) * len(train_loader)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            use_mixup=(not args.no_mixup),
            writer=writer, global_step_offset=batch_offset,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s) LR={current_lr:.2e}")
        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")

        # TensorBoard
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("learning_rate", current_lr, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  ** New best: {val_acc:.2f}% **")
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

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    model.eval()

    use_tta = not args.no_tta
    print(f"Test-time augmentation: {'ON (5 views)' if use_tta else 'OFF'}")

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
                with autocast("cuda", enabled=(device.type == "cuda")):
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

    if use_tta:
        _, test_acc_no_tta = evaluate(model, test_loader, criterion, device, use_tta=False)
        print(f"\n  (Without TTA:   {test_acc_no_tta:.2f}%)")

    # TensorBoard: final metrics
    writer.add_scalar("test/accuracy", test_acc, args.epochs)
    writer.add_scalar("test/sensitivity", sensitivity, args.epochs)
    writer.add_scalar("test/specificity", specificity, args.epochs)
    writer.add_scalar("test/precision", precision, args.epochs)
    writer.add_scalar("test/f1", f1, args.epochs)

    # Confusion matrix figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        cm = np.array([[tp, fn], [fp, tn]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted +", "Predicted -"])
        ax.set_yticklabels(["Actual +", "Actual -"])
        ax.set_title(f"Test Confusion Matrix (Acc: {test_acc:.2f}%)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
        fig.tight_layout()
        writer.add_figure("test/confusion_matrix", fig, args.epochs)
        plt.close(fig)
    except ImportError:
        pass

    # Sample predictions
    try:
        sample_images, sample_labels = next(iter(test_loader))
        sample_images = sample_images.to(device)
        with torch.inference_mode():
            sample_probs = torch.softmax(model(sample_images), dim=1)
            _, sample_preds = sample_probs.max(1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        display_imgs = sample_images[:16].cpu() * std + mean
        display_imgs = display_imgs.clamp(0, 1)

        grid = make_grid(display_imgs, nrow=4, normalize=False)
        writer.add_image("test/sample_predictions", grid, args.epochs)
    except Exception:
        pass

    # HParams
    writer.add_hparams(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "drop_path": args.drop_path,
            "dropout": args.dropout,
            "mixup": not args.no_mixup,
            "tta": not args.no_tta,
        },
        {
            "hparam/test_accuracy": test_acc,
            "hparam/best_val_accuracy": best_val_acc,
            "hparam/test_f1": f1,
            "hparam/test_sensitivity": sensitivity,
        },
    )

    # --- Save model ---
    from core.model_manager import get_model_manager
    mm = get_model_manager()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        model_dir = mm.ensure_model_dir("malaria-plasmosenet")
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
