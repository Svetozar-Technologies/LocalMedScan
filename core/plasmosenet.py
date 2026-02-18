"""PlasmoSENet — Custom CNN for malaria parasite detection.

A ResNet-style architecture designed specifically for blood smear microscopy:
  - Multi-scale stem (3x3 + 5x5 + 7x7) captures parasites at different sizes
  - Squeeze-and-Excitation channel attention for color-aware feature weighting
  - Stochastic depth (drop path) for regularization when training from scratch
  - Kaiming initialization for stable from-scratch convergence

Architecture: ~10.6M parameters, ~42 MB model file.
Trained from scratch on the NIH Malaria Cell Images Dataset.

Interface compatible with torchvision models:
  - model.features  : nn.Sequential of conv stages
  - model.classifier: nn.Sequential ending with Linear
  - model.last_channel: int (384)
  - model.features[-1]: conv block (for GradCAM)
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Learns to weight feature channels by their relevance. Critical for
    malaria detection where purple/blue parasite stain channels carry
    more signal than pink/red erythrocyte background channels.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DropPath(nn.Module):
    """Stochastic depth: drops entire residual branch during training.

    Linearly increases drop probability with network depth, providing
    implicit ensemble regularization. Disabled at inference time.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype)
        mask = torch.floor_(mask + keep_prob)
        return x * mask / keep_prob


class MalariaResBlock(nn.Module):
    """Residual block with SE attention and stochastic depth.

    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> SE -> DropPath -> + shortcut -> ReLU

    Uses 1x1 conv shortcut when channels or spatial dims change.
    """

    def __init__(self, in_ch, out_ch, stride=1, se_reduction=16, drop_path_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)
        out = self.relu(out + identity)
        return out


class MultiScaleStem(nn.Module):
    """Multi-scale stem with parallel 3x3, 5x5, and 7x7 convolutions.

    Designed for malaria images where parasites appear at varying sizes:
      - Ring forms (~2-3 um) -> captured by 3x3 branch
      - Trophozoites (~4-6 um) -> captured by 5x5 branch
      - Schizonts (~6-8 um) -> captured by 7x7 branch

    Branches are concatenated and fused with a 1x1 conv.
    Output: (B, out_ch, H/2, W/2)
    """

    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        branch_ch = out_ch // 4  # 16 channels per branch

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        return self.fuse(torch.cat([b3, b5, b7], dim=1))


class PlasmoSENet(nn.Module):
    """PlasmoSENet: Custom CNN for malaria parasite detection from blood smear images.

    Architecture:
        Stem (MultiScale 3->64) -> Stage1 (64, 2 blocks) -> Stage2 (128, 3 blocks)
        -> Stage3 (256, 4 blocks) -> Stage4 (384, 2 blocks) -> Head Conv
        -> GAP -> Dropout -> Linear(384, num_classes)

    Args:
        num_classes: Number of output classes (default: 2).
        drop_path_rate: Maximum stochastic depth rate (default: 0.2).
        dropout: Dropout rate before classifier (default: 0.3).
    """

    def __init__(self, num_classes=2, drop_path_rate=0.2, dropout=0.3):
        super().__init__()
        self.last_channel = 384

        # Linearly increasing drop path rates across 11 residual blocks
        num_blocks = 11
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]

        features = []

        # Index 0: Multi-scale stem (3 -> 64, spatial /2)
        features.append(MultiScaleStem(in_ch=3, out_ch=64))

        # Index 1-2: Stage 1 — 64 channels, 112x112
        features.append(MalariaResBlock(64, 64, stride=1, drop_path_rate=dpr[0]))
        features.append(MalariaResBlock(64, 64, stride=1, drop_path_rate=dpr[1]))

        # Index 3-5: Stage 2 — 128 channels, 56x56
        features.append(MalariaResBlock(64, 128, stride=2, drop_path_rate=dpr[2]))
        features.append(MalariaResBlock(128, 128, stride=1, drop_path_rate=dpr[3]))
        features.append(MalariaResBlock(128, 128, stride=1, drop_path_rate=dpr[4]))

        # Index 6-9: Stage 3 — 256 channels, 28x28
        features.append(MalariaResBlock(128, 256, stride=2, drop_path_rate=dpr[5]))
        features.append(MalariaResBlock(256, 256, stride=1, drop_path_rate=dpr[6]))
        features.append(MalariaResBlock(256, 256, stride=1, drop_path_rate=dpr[7]))
        features.append(MalariaResBlock(256, 256, stride=1, drop_path_rate=dpr[8]))

        # Index 10-11: Stage 4 — 384 channels, 14x14
        features.append(MalariaResBlock(256, 384, stride=2, drop_path_rate=dpr[9]))
        features.append(MalariaResBlock(384, 384, stride=1, drop_path_rate=dpr[10]))

        # Index 12: Head conv block — GradCAM target (features[-1])
        features.append(nn.Sequential(
            nn.Conv2d(384, 384, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Kaiming initialization for training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
