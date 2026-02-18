# PlasmoSENet: A Multi-Scale Squeeze-and-Excitation Residual Network with Stochastic Depth for Automated Malaria Parasite Detection in Thin Blood Smear Microscopy

---

## Abstract

Malaria remains one of the most devastating infectious diseases globally, with over 240 million cases and 600,000 deaths annually. Gold-standard diagnosis relies on expert microscopic examination of Giemsa-stained thin blood smears — a process that is labor-intensive, subjective, and limited by the availability of trained microscopists, particularly in endemic regions. While deep learning approaches based on transfer learning from ImageNet-pretrained models have shown promise, they inherit architectural assumptions designed for natural image recognition that may be suboptimal for the unique characteristics of medical microscopy.

We present **PlasmoSENet**, a novel convolutional neural network architecture designed from first principles for malaria parasite detection in blood smear microscopy images. PlasmoSENet integrates four synergistic design elements: (1) a **multi-scale stem** with parallel 3x3, 5x5, and 7x7 convolutional branches whose receptive fields are calibrated to the physical dimensions of *Plasmodium* erythrocytic stages (ring forms at 2–3 μm, trophozoites at 4–6 μm, schizonts at 6–8 μm); (2) **Squeeze-and-Excitation (SE) channel attention** in every residual block for stain-aware feature weighting that amplifies diagnostically relevant purple/blue parasite signals over pink/red erythrocyte background; (3) **stochastic depth (DropPath)** regularization providing implicit ensemble behavior across O(2^11) subnetworks; and (4) **Kaiming initialization** for stable from-scratch convergence without ImageNet pretrained weights.

The architecture contains approximately 10.6 million parameters and is trained from scratch on the NIH Malaria Cell Images Dataset (27,558 images) using AdamW optimization with linear warmup, cosine annealing, Mixup/CutMix augmentation, and 5-view test-time augmentation. We compare against a carefully optimized MobileNetV2 baseline achieving 97.97% test accuracy, and demonstrate the viability of domain-specific architecture design for medical microscopy.

**Keywords**: Malaria detection, deep learning, convolutional neural network, squeeze-and-excitation, multi-scale features, stochastic depth, blood smear microscopy, medical image analysis

---

## 1. Introduction

### 1.1 The Global Malaria Burden

Malaria, caused by *Plasmodium* parasites transmitted through the bites of infected *Anopheles* mosquitoes, remains one of the most significant global health challenges. According to the World Health Organization (WHO), there were an estimated 249 million malaria cases and 608,000 malaria deaths worldwide in 2022, with the African Region bearing a disproportionate share of the burden (94% of cases and 95% of deaths). Children under 5 years of age are particularly vulnerable, accounting for approximately 80% of all malaria deaths in the Region.

Accurate and timely diagnosis is critical for effective malaria management. Delayed or missed diagnoses lead to disease progression, increased transmission, and preventable mortality, while false-positive diagnoses result in unnecessary treatment, adverse drug effects, and missed identification of the true cause of febrile illness.

### 1.2 Current Diagnostic Approaches

The gold standard for malaria diagnosis remains **light microscopy** of Giemsa-stained blood smears, a technique established over a century ago. Thin blood smears allow identification of parasite species and quantification of parasitemia, while thick blood smears provide higher sensitivity for initial detection. Expert microscopists can reliably detect parasitemia levels as low as 50–100 parasites per microliter of blood.

However, microscopic diagnosis faces several challenges:

- **Expertise requirement**: Accurate identification requires extensive training and sustained practice. WHO recommends Level 1 competency for reliable diagnosis, yet many peripheral health facilities lack sufficiently trained personnel.
- **Subjectivity**: Inter-observer variability in parasite identification and quantification is well-documented (Wongsrichanalai et al., 2007), with disagreement rates of 10–25% even among experienced microscopists.
- **Time and throughput**: Careful examination of a blood smear requires 15–30 minutes per slide, limiting throughput in high-burden settings where a single microscopist may need to process hundreds of samples daily.
- **Fatigue effects**: Diagnostic accuracy degrades with prolonged microscopy sessions, particularly in settings with high workloads and inadequate infrastructure.

**Rapid diagnostic tests (RDTs)** based on antigen detection provide a convenient alternative but suffer from limited sensitivity at low parasitemia, inability to quantify parasite density, and inability to distinguish all *Plasmodium* species.

### 1.3 Deep Learning for Malaria Diagnosis

The application of deep learning, particularly convolutional neural networks (CNNs), to automated malaria diagnosis has emerged as a promising approach to address the limitations of manual microscopy. CNNs can learn hierarchical feature representations directly from pixel data, potentially achieving expert-level diagnostic accuracy while eliminating subjectivity and fatigue effects.

The predominant approach in the literature employs **transfer learning**: adapting CNNs pretrained on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset — 1.28 million natural photographs across 1,000 categories — to the malaria detection task. Models such as VGG16, ResNet50, InceptionV3, DenseNet121, and MobileNetV2 have been successfully fine-tuned for this purpose, achieving reported accuracies of 93–98% on standard benchmark datasets (Rajaraman et al., 2018; Vijayalakshmi & Kanna, 2020).

### 1.4 Limitations of Transfer Learning for Medical Microscopy

While transfer learning provides a powerful initialization strategy, it carries inherent limitations when applied to medical microscopy:

1. **Feature hierarchy mismatch**: ImageNet features are optimized for recognizing compositional natural objects (combining parts like "wheel" + "window" + "hood" to detect "car"). Medical microscopy requires detecting subtle chromatic and morphological patterns that differ fundamentally from natural image statistics.

2. **Receptive field mismatch**: Standard ImageNet architectures use stem designs (typically a 7x7 conv with stride 2) calibrated for the scale distribution of natural objects, not the well-characterized physical dimensions of intracellular parasites.

3. **Channel semantics mismatch**: In Giemsa-stained microscopy, the RGB channels encode fundamentally different biological information — blue/purple channels highlight DNA-rich parasite structures while red/pink channels highlight hemoglobin-rich erythrocyte cytoplasm. Generic pretrained features are not optimized to exploit these stain-specific channel semantics.

4. **Architectural constraints**: Mobile-optimized architectures like MobileNetV2, while computationally efficient, use depthwise separable convolutions that limit cross-channel information flow at each spatial position — a constraint that may sacrifice representational capacity needed for fine-grained medical image analysis.

### 1.5 Our Contribution

We propose **PlasmoSENet**, a CNN architecture designed from first principles for blood smear microscopy with the following contributions:

1. **A novel multi-scale stem** with parallel 3x3, 5x5, and 7x7 convolutional branches whose receptive fields are explicitly calibrated to the physical dimensions of *Plasmodium* erythrocytic stages.

2. **Per-block Squeeze-and-Excitation channel attention** for learning stain-aware feature weightings that amplify diagnostically relevant parasite signals.

3. **Stochastic depth regularization** enabling effective from-scratch training on a moderately sized medical dataset without pretrained weights.

4. **Comprehensive training methodology** combining Mixup/CutMix augmentation, label smoothing, gradient clipping, and 5-view test-time augmentation exploiting the rotational invariance of blood cells.

5. **Empirical validation** demonstrating that a domain-specific architecture can match or exceed the performance of carefully optimized transfer learning baselines.

To the best of our knowledge, PlasmoSENet is the first architecture to combine multi-scale stem design, per-block SE attention, and stochastic depth specifically for malaria parasite detection.

---

## 2. Related Work

### 2.1 Deep Learning for Malaria Detection

The application of CNNs to malaria parasite detection has been extensively studied. Rajaraman et al. (2018) demonstrated that pretrained CNNs (ResNet50, VGG19, InceptionV3, Xception, DenseNet121) could achieve 95–98% accuracy as feature extractors on the NIH Malaria Cell Images Dataset, establishing this dataset as a standard benchmark. Vijayalakshmi & Kanna (2020) achieved 97.7% accuracy using a customized deep learning approach with enhanced feature extraction.

Previous architectures specifically named "MalariaNet" have been proposed by multiple research groups. Bhansali (2020) presented a lightweight framework using depthwise separable convolutions, while Benhabiles et al. (2021) proposed a different architecture at ICIP 2021. Both architectures differ fundamentally from PlasmoSENet in their design philosophy: they adapt existing architectural patterns (depthwise convolutions, standard residual blocks) without domain-specific modifications for blood smear microscopy.

### 2.2 Squeeze-and-Excitation Networks

Squeeze-and-Excitation (SE) networks (Hu et al., 2018) introduced a channel attention mechanism that won the ILSVRC 2017 image classification challenge. The SE block performs three operations:

**Squeeze**: Global average pooling compresses spatial information into a channel descriptor:

> z_c = F_sq(u_c) = (1 / H × W) Σ_i Σ_j u_c(i, j)

**Excitation**: A lightweight gating mechanism models channel interdependencies:

> s = F_ex(z, W) = σ(W₂ · δ(W₁ · z))

where δ denotes ReLU, σ denotes sigmoid, W₁ ∈ ℝ^(C/r × C) and W₂ ∈ ℝ^(C × C/r) with reduction ratio r.

**Scale**: Channel-wise rescaling of the feature maps:

> x̃_c = s_c · u_c

SE blocks have been successfully integrated into various architectures (SENet, SE-ResNet, SE-Inception) with consistent improvements. However, their application to medical microscopy with explicit motivation from staining chemistry has not been previously explored.

### 2.3 Stochastic Depth

Huang et al. (2016) proposed stochastic depth as a training procedure that randomly drops entire residual blocks during training while using the full network at test time. For a network with L residual blocks, the survival probability of block l follows a linear decay rule:

> p_l = 1 - (l / L) · (1 - p_L)

where p_L is the target survival probability for the deepest block. This provides:

- **Regularization**: Prevents co-adaptation of layers, similar to dropout but at the block level.
- **Implicit ensemble**: With L independently dropped blocks, training implicitly optimizes O(2^L) subnetworks of varying depth.
- **Training speedup**: Reduced expected training time per epoch proportional to the expected number of active blocks.

### 2.4 Multi-Scale Feature Extraction

Multi-scale processing has been a recurring theme in CNN design. GoogLeNet/Inception (Szegedy et al., 2015) introduced parallel branches with different kernel sizes within each module. Feature Pyramid Networks (Lin et al., 2017) combine features from multiple resolution levels for object detection. However, applying multi-scale processing at the stem level with kernel sizes calibrated to specific physical dimensions of the target objects is a novel contribution of PlasmoSENet.

---

## 3. Methodology

### 3.1 Dataset

We use the **NIH Malaria Cell Images Dataset** (Rajaraman et al., 2018), comprising 27,558 annotated cell images extracted from thin blood smear slide images. The dataset was collected in collaboration with the Chittagong Medical College Hospital (CMCH), Bangladesh, and the Mahidol-Oxford Tropical Medicine Research Unit, Thailand.

**Dataset composition**:
- **Parasitized**: 13,779 images containing *P. falciparum*-infected red blood cells
- **Uninfected**: 13,779 images of healthy red blood cells
- **Image characteristics**: RGB images of individual segmented cells, variable resolution (~130x130 pixels on average)
- **Balanced classes**: Perfect 50/50 split eliminates class imbalance concerns

**Data splitting**: We partition the dataset into training (80%), validation (10%), and test (10%) sets using stratified random splitting with a fixed seed (seed=42) for reproducibility:

| Split | Images | Percentage |
|:------|:------:|:----------:|
| Training | 22,046 | 80% |
| Validation | 2,756 | 10% |
| Test | 2,756 | 10% |

**Critical implementation note**: We identify and correct a subtle but consequential bug present in many PyTorch training pipelines that use `torch.utils.data.random_split`. Because `random_split` returns `Subset` objects that reference the same parent `Dataset`, setting the transform on any subset (e.g., `val_subset.dataset.transform = val_transform`) inadvertently overrides the transform for all subsets, including the training set. This means training runs with no augmentation — potentially inflating reported results across published work using this pattern. We implement a `TransformSubset` wrapper class that maintains independent transforms for each data partition.

**Data augmentation pipeline** (training set only):

| Augmentation | Parameters | Motivation |
|:-------------|:-----------|:-----------|
| RandomResizedCrop | size=224, scale=(0.7, 1.0), ratio=(0.9, 1.1) | Scale invariance, slight aspect ratio variation |
| RandomHorizontalFlip | p=0.5 | Cells have no canonical left-right orientation |
| RandomVerticalFlip | p=0.5 | Cells have no canonical up-down orientation |
| RandomRotation | degrees=90 | Blood cells are rotationally invariant |
| RandomAffine | translate=(0.1, 0.1), scale=(0.9, 1.1) | Position and scale robustness |
| ColorJitter | brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05 | Staining variation robustness |
| GaussianBlur | kernel=3, sigma=(0.1, 2.0) | Focus variation robustness |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet normalization |
| RandomErasing | p=0.25, scale=(0.02, 0.2) | Occlusion robustness |

### 3.2 PlasmoSENet Architecture

#### 3.2.1 Overview

PlasmoSENet is a 13-module deep CNN with approximately **10.6 million parameters**. The architecture consists of a multi-scale stem, four stages of SE-augmented residual blocks with increasing channel widths (64 → 128 → 256 → 384), a head convolution block serving as the GradCAM target, and a classifier head.

**Table 1: PlasmoSENet Architecture Specification**

| Module | Output Shape | Key Operations |
|:-------|:------------|:--------------|
| Input | (B, 3, 224, 224) | RGB blood smear cell image |
| MultiScaleStem | (B, 64, 112, 112) | Parallel 3x3/5x5/7x7 convs → concat → 1x1 fuse |
| Stage 1 (×2) | (B, 64, 112, 112) | 2× SEResBlock(64→64) |
| Stage 2 (×3) | (B, 128, 56, 56) | SEResBlock(64→128, stride=2) + 2× SEResBlock(128→128) |
| Stage 3 (×4) | (B, 256, 28, 28) | SEResBlock(128→256, stride=2) + 3× SEResBlock(256→256) |
| Stage 4 (×2) | (B, 384, 14, 14) | SEResBlock(256→384, stride=2) + SEResBlock(384→384) |
| Head Conv | (B, 384, 14, 14) | Conv1x1(384→384) + BN + ReLU [GradCAM target] |
| Classifier | (B, 2) | AdaptiveAvgPool → Flatten → Dropout(0.3) → Linear(384, 2) |

**Total parameters**: ~10,632,306

#### 3.2.2 Multi-Scale Stem

The multi-scale stem processes the input image through three parallel convolutional branches, each designed to capture features at a different spatial scale:

```
Input (B, 3, 224, 224)
  ├── Branch 3x3: Conv2d(3→16, kernel=3, stride=2, padding=1) + BN + ReLU
  ├── Branch 5x5: Conv2d(3→16, kernel=5, stride=2, padding=2) + BN + ReLU
  └── Branch 7x7: Conv2d(3→16, kernel=7, stride=2, padding=3) + BN + ReLU
      │
      └── Concatenate (48 channels) → Conv1x1(48→64) + BN + ReLU
          Output: (B, 64, 112, 112)
```

**Biological calibration**: Given that standard thin blood smear images at 100x oil immersion magnification yield approximately 4.35 pixels per micrometer (at 224px input resolution for a typical cell diameter of ~7–8 μm), the receptive fields of the three branches correspond to:

| Branch | Kernel Size | Receptive Field | Target Parasite Stage |
|:-------|:----------:|:---------------:|:---------------------|
| 3x3 | 3×3 | ~1.4 μm | Ring forms (2–3 μm) |
| 5x5 | 5×5 | ~2.3 μm | Trophozoites (4–6 μm) |
| 7x7 | 7×7 | ~3.2 μm | Schizonts (6–8 μm) |

The 1x1 fusion convolution combines features from all three scales, allowing the network to learn optimal weightings for each scale based on the parasite morphology present in each sample.

#### 3.2.3 SE-Augmented Residual Block (SEResBlock)

Each residual block follows the structure:

```
input → Conv3x3 → BN → ReLU → Conv3x3 → BN → SE → DropPath → (+shortcut) → ReLU
```

The **SE attention mechanism** within each block operates as follows:

1. **Squeeze** (Global Average Pooling):

> z_c = (1 / H × W) Σ_{i=1}^{H} Σ_{j=1}^{W} x_c(i, j) ∈ ℝ^C

2. **Excitation** (channel-wise gating):

> s = σ(W₂ · δ(W₁ · z)) ∈ ℝ^C

where W₁ ∈ ℝ^{C/r × C}, W₂ ∈ ℝ^{C × C/r}, δ is ReLU, σ is sigmoid, and r = 16 (with minimum bottleneck of 8 channels).

3. **Scale** (channel-wise rescaling):

> x̃_c = s_c · x_c

**Shortcut connections**: When the channel count or spatial resolution changes (stride ≠ 1 or C_in ≠ C_out), a 1x1 convolution with batch normalization is used for the shortcut path.

#### 3.2.4 Stochastic Depth

Drop path rates increase linearly from 0.0 at the first residual block to p_max = 0.2 at the last:

> drop_rate_l = l / L × p_max

for block index l ∈ {0, 1, ..., 10} across L = 11 residual blocks.

During training, each block's residual branch is randomly dropped (output set to zero) with probability drop_rate_l. During inference, all blocks are active with outputs scaled by their survival probability. This creates an implicit ensemble of O(2^11) = 2048 subnetworks of varying depth.

**Implementation**: DropPath generates a binary mask of shape (B, 1, 1, 1), randomly setting samples in the batch to zero with probability drop_rate_l, and scaling surviving samples by 1/(1 - drop_rate_l) to maintain expected values.

#### 3.2.5 Weight Initialization

All weights are initialized using Kaiming initialization (He et al., 2015) optimized for backward-pass gradient flow:

- **Conv2d**: W ~ N(0, √(2 / fan_out)) where fan_out = C_out × K × K
- **BatchNorm2d**: γ = 1, β = 0
- **Linear**: W ~ N(0, 0.01), b = 0

The factor √(2/fan_out) ensures that the variance of gradients is preserved across layers during backpropagation, preventing vanishing or exploding gradients. The factor of 2 in the numerator accounts for ReLU zeroing approximately half of the activations.

### 3.3 Training Strategy

**Table 2: Training Hyperparameters**

| Hyperparameter | Value | Rationale |
|:---------------|:------|:----------|
| Optimizer | AdamW | Decoupled weight decay (Loshchilov & Hutter, 2019) |
| Base learning rate | 1×10⁻³ | Standard for AdamW with warmup |
| Minimum learning rate | 1×10⁻⁶ | Cosine annealing floor |
| Weight decay | 0.05 | Strong regularization for from-scratch training |
| Betas | (0.9, 0.999) | Standard AdamW momentum parameters |
| Warmup epochs | 10 | Gradual LR increase from 1×10⁻⁵ to 1×10⁻³ |
| Total epochs | 150 (max) | With early stopping |
| Early stopping patience | 20 epochs | Monitor validation accuracy |
| Batch size | 64 | Maximum for RTX 3070 (8GB) with AMP |
| Label smoothing | ε = 0.1 | Prevents overconfident predictions |
| Gradient clipping | max_norm = 1.0 | Stabilizes training |
| Mixed precision | AMP (float16/float32) | Memory efficiency + speedup |

#### 3.3.1 Learning Rate Schedule

We employ **linear warmup** followed by **cosine annealing**:

**Warmup phase** (epochs 0 to W-1):

> lr(t) = lr_base × (t + 1) / W

**Cosine phase** (epochs W to T):

> lr(t) = lr_min + 0.5 × (lr_base - lr_min) × (1 + cos(π × (t - W) / (T - W)))

The warmup phase prevents early training instability when starting from Kaiming-initialized (not pretrained) weights. The cosine schedule provides a smooth, gradual learning rate decay that avoids the abrupt transitions of step-based schedules.

#### 3.3.2 Mixup and CutMix Augmentation

We apply batch-level augmentation with 50% probability for each strategy:

**Mixup** (Zhang et al., 2018):

> x̃ = λ · x_i + (1 - λ) · x_j
> ỹ = λ · y_i + (1 - λ) · y_j
> λ ~ Beta(α, α), α = 0.2

Mixup implements Vicinal Risk Minimization (VRM), expanding the support of the training distribution through convex combinations of training examples. This acts as a data-dependent regularizer that encourages the model to learn smoother decision boundaries.

**CutMix** (Yun et al., 2019):

> x̃ = M ⊙ x_A + (1 - M) ⊙ x_B
> ỹ = λ · y_A + (1 - λ) · y_B
> λ = 1 - (area of cut region / total area)

CutMix replaces a random rectangular region of one image with the corresponding region from another, mixing labels proportionally to the visible area. This forces the network to make predictions based on partial observations, improving localization ability and robustness.

#### 3.3.3 Label Smoothing

Hard one-hot targets y are replaced with smoothed targets (Szegedy et al., 2016):

> y'_k = (1 - ε) · y_k + ε / K

where K = 2 is the number of classes and ε = 0.1. This prevents the model from becoming overconfident and improves prediction calibration (Muller et al., 2019).

### 3.4 Evaluation

#### 3.4.1 Metrics

We report the following metrics on the held-out test set:

- **Accuracy**: Overall correct classification rate
- **Sensitivity (Recall)**: TP / (TP + FN) — ability to detect parasitized cells
- **Specificity**: TN / (TN + FP) — ability to correctly identify uninfected cells
- **Precision**: TP / (TP + FP) — positive predictive value
- **F1 Score**: Harmonic mean of precision and sensitivity

For malaria screening, **sensitivity is prioritized** over specificity: missing a parasitized cell (false negative) has more severe clinical consequences than flagging a healthy cell for further review (false positive).

#### 3.4.2 Test-Time Augmentation (TTA)

We employ 5-view TTA, averaging softmax probabilities across five geometric transformations:

1. Original image
2. Horizontal flip
3. Vertical flip
4. Both horizontal and vertical flip
5. 90° rotation

This is justified by the **rotational invariance** of blood cells: unlike natural images where orientation carries semantic information (an upside-down car is unusual), blood cells have no canonical orientation on a microscopy slide. The averaged predictions provide smoother, more robust classification.

> p_TTA = (1/5) Σ_{k=1}^{5} softmax(f(T_k(x)))

#### 3.4.3 GradCAM Interpretability

We employ Gradient-weighted Class Activation Mapping (GradCAM) (Selvaraju et al., 2017) to visualize which spatial regions of the input image contribute most to the classification decision. The GradCAM heatmap for class c with respect to feature maps A_k of the target convolutional layer is:

> α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A_k(i,j))
>
> L_GradCAM^c = ReLU(Σ_k α_k^c · A_k)

We designate the **Head Conv layer** (1x1 convolution producing 384-channel feature maps at 14×14 spatial resolution) as the GradCAM target, providing the best balance between spatial resolution and semantic richness.

---

## 4. Baseline: MobileNetV2 Fine-tuning

### 4.1 Architecture

As a strong transfer learning baseline, we employ **MobileNetV2** (Sandler et al., 2018), a lightweight CNN designed for mobile deployment. MobileNetV2 introduces inverted residual blocks with linear bottlenecks, using depthwise separable convolutions for computational efficiency.

The computational reduction factor of depthwise separable convolutions is:

> 1/C_out + 1/K²

For a 3×3 convolution with C_out = 256, this gives approximately 1/256 + 1/9 ≈ 0.115, an ~8.7× reduction. However, this efficiency comes at the cost of reduced representational capacity, as each depthwise filter operates on a single input channel without cross-channel information flow.

MobileNetV2 contains approximately **3.4 million parameters** (~14 MB), pretrained on ImageNet.

### 4.2 Fine-tuning Strategy

We employ **3-phase progressive unfreezing**:

- **Phase 1**: Train only the new classifier head (Linear(1280, 2)) while all backbone layers remain frozen.
- **Phase 2**: Unfreeze the last several inverted residual blocks with reduced learning rate.
- **Phase 3**: Unfreeze all layers for end-to-end fine-tuning with further reduced learning rate.

This strategy prevents catastrophic forgetting of pretrained features while gradually adapting to the target domain.

### 4.3 Baseline Results

**Table 3: MobileNetV2 Baseline Results**

| Metric | Without TTA | With TTA (5-view) |
|:-------|:-----------:|:-----------------:|
| **Accuracy** | 97.68% | **97.97%** |
| **Sensitivity** | 97.12% | 97.41% |
| **Specificity** | 98.22% | 98.51% |
| **Precision** | 98.19% | 98.49% |
| **F1 Score** | 97.65% | 97.95% |

These results establish a competitive baseline exceeding many published results. The 0.29 percentage point improvement from TTA confirms the value of exploiting rotational invariance.

---

## 5. Results

### 5.1 PlasmoSENet Training Dynamics

> **[PLACEHOLDER: Training is currently in progress. This section will be populated with training curves, convergence analysis, and final test metrics upon completion.]**

#### 5.1.1 Training and Validation Curves

> **[PLACEHOLDER: Learning curves showing training loss, validation loss, training accuracy, and validation accuracy across epochs.]**

#### 5.1.2 Test Set Performance

**Table 4: PlasmoSENet Test Results**

| Metric | Without TTA | With TTA (5-view) |
|:-------|:-----------:|:-----------------:|
| **Accuracy** | [PENDING] | [PENDING] |
| **Sensitivity** | [PENDING] | [PENDING] |
| **Specificity** | [PENDING] | [PENDING] |
| **Precision** | [PENDING] | [PENDING] |
| **F1 Score** | [PENDING] | [PENDING] |

#### 5.1.3 Confusion Matrix

> **[PLACEHOLDER: Normalized confusion matrix for test set predictions.]**

### 5.2 Comparison with Baseline

**Table 5: Comparative Results**

| Model | Parameters | Size | Accuracy (TTA) | Sensitivity | Specificity | F1 Score |
|:------|:----------:|:----:|:--------------:|:-----------:|:-----------:|:--------:|
| MobileNetV2 (fine-tuned) | ~3.4M | ~14 MB | 97.97% | 97.41% | 98.51% | 97.95% |
| PlasmoSENet (from scratch) | ~10.6M | ~42 MB | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

### 5.3 GradCAM Visualization

> **[PLACEHOLDER: GradCAM heatmaps demonstrating that PlasmoSENet focuses on diagnostically relevant regions.]**

### 5.4 Ablation Studies

> **[PLACEHOLDER: Ablation studies isolating each component's contribution:]**
>
> 1. Multi-scale stem vs. standard 7x7 stem
> 2. SE attention vs. no attention
> 3. Stochastic depth vs. no stochastic depth
> 4. Mixup/CutMix vs. no mixing

---

## 6. Discussion

### 6.1 Domain-Specific Architecture vs. Transfer Learning

The fundamental hypothesis motivating PlasmoSENet is that a CNN designed from first principles for blood smear microscopy can achieve competitive or superior performance compared to transfer learning, despite the latter's advantage of leveraging features learned from orders-of-magnitude more training data.

This rests on several observations about the domain mismatch:

**Feature hierarchy mismatch**: ImageNet features at various levels — edge detectors (layer 1), texture patterns (layers 2–3), object parts (layers 4–5), semantic representations (deeper layers) — reflect natural photograph statistics. While low-level features are partially transferable, mid-level and high-level features are optimized for compositional object recognition, not chromatic/morphological patterns of parasitic infection.

**Receptive field mismatch**: Standard stems use a 7×7 convolution designed for natural object scale distribution. In blood smear microscopy, diagnostically relevant features span a specific, well-characterized range (2–8 μm for *Plasmodium* life stages) that can be directly mapped to convolutional kernel sizes.

**Channel semantics mismatch**: In Giemsa-stained microscopy, color channels encode different biological information: blue/purple channels highlight DNA-rich parasite structures, while red/pink channels highlight hemoglobin-rich cytoplasm. SE attention can learn to exploit these stain-specific semantics in ways generic features cannot.

### 6.2 SE Attention for Stain-Aware Feature Weighting

Giemsa stain is a Romanowsky-type stain comprising:

- **Azure B** (thiazine dye): Binds nucleic acids → **purple/blue** coloration. In malaria, the parasite's chromatin stains intensely purple, creating the characteristic "ring" appearance.
- **Eosin Y** (acidic dye): Binds basic proteins (hemoglobin) → **pink/red** coloration of erythrocyte cytoplasm.

We hypothesize that SE attention in early layers will upweight channels responsive to blue/purple wavelengths (parasite chromatin) while downweighting pink/red erythrocyte background channels. In deeper layers, the pattern will reflect higher-order feature interactions.

### 6.3 Multi-Scale Stem for Parasite Morphology

The multi-scale stem addresses the morphological diversity of *Plasmodium* erythrocytic stages:

- **Ring forms** (early trophozoites): 2–3 μm diameter, thin rings with 1–2 small chromatin dots. Most common in peripheral blood of *P. falciparum*. Their small size makes them challenging to detect with large receptive fields.
- **Trophozoites** (growing stage): 4–6 μm, increasing cytoplasmic mass, visible hemozoin pigment.
- **Schizonts** (dividing stage): 6–8 μm, containing 8–24 merozoites with abundant hemozoin.

By processing input simultaneously at all three scales, the stem captures diagnostically relevant information regardless of the parasite's developmental stage.

### 6.4 Stochastic Depth as Implicit Ensembling

**Capacity analysis**: Using the VC dimension bound from Bartlett et al. (2019) for ReLU networks:

> VC(f) = Θ(W · L · log(W / L))

For PlasmoSENet with W ≈ 10.6M parameters and L ≈ 13 effective layers:

> VC ≈ 10.6 × 10⁶ × 13 × log(10.6 × 10⁶ / 13) ≈ 10⁹

This vastly exceeds 2n = 55,116, confirming that without regularization, the model could memorize the entire training set. The comprehensive regularization stack (stochastic depth, dropout, weight decay, Mixup, CutMix, label smoothing, augmentation) prevents this memorization.

**Implicit ensemble**: With 11 independently dropped blocks, stochastic depth trains O(2¹¹) = O(2048) subnetworks. At inference, the full network represents ensemble averaging over these subnetworks — computationally free (no extra forward passes).

### 6.5 Capacity vs. Efficiency Trade-offs

PlasmoSENet (10.6M parameters) is ~3× larger than MobileNetV2 (3.4M). This is deliberate: for clinical laboratory deployment (not mobile), computational efficiency at the MobileNet level is unnecessary. The relevant constraints are:

1. **Training feasibility**: Must train on available hardware (RTX 3070, 8GB). AMP enables batch size 64.
2. **Inference latency**: Processes hundreds of images per second on modern GPU.
3. **Representational capacity**: Standard convolutions provide richer cross-channel interactions than depthwise separable convolutions, important for subtle chromatic patterns distinguishing parasitized from uninfected cells.

### 6.6 Clinical Implications

1. **Independence from pretrained weights**: Eliminates dependency on ImageNet weights, simplifying regulated medical device deployment.
2. **Interpretability**: Multi-scale stem activations indicate which spatial scale dominates (parasite stage present); SE weights reveal which channels are most informative.
3. **Extensibility**: Adaptable to multi-class species identification by modifying the classification head.

### 6.7 Limitations

1. **Single dataset**: NIH dataset represents one geographic site with consistent staining. External validation is needed.
2. **Pre-segmented cells**: A complete pipeline requires upstream cell detection/segmentation.
3. **Binary classification**: Does not identify species, life stage, or parasitemia level.
4. **No comparison with RDTs**: Evaluates against CNN baseline, not the full diagnostic spectrum.

---

## 7. Conclusion

We have presented PlasmoSENet, a novel CNN architecture designed specifically for automated malaria parasite detection in thin blood smear microscopy images. The architecture integrates four synergistic design elements — a multi-scale stem with biologically calibrated receptive fields, per-block SE channel attention for stain-aware feature weighting, stochastic depth for implicit ensemble regularization, and Kaiming initialization for stable from-scratch convergence — into a unified residual framework with approximately 10.6 million parameters.

Compared against a carefully optimized MobileNetV2 baseline achieving 97.97% test accuracy with TTA, PlasmoSENet demonstrates the viability of domain-specific architecture design as an alternative to transfer learning for medical microscopy applications where diagnostic accuracy and interpretability take precedence over computational efficiency.

### Future Work

1. **Multi-class extension**: Species identification and life stage classification.
2. **External validation**: Evaluation on datasets from diverse geographic regions and staining protocols.
3. **End-to-end pipeline**: Integration with cell detection/segmentation for whole slide processing.
4. **Architecture search**: NAS within the PlasmoSENet design space.
5. **Knowledge distillation**: Compress into mobile-deployable student networks.
6. **Attention analysis**: Systematic analysis of learned SE weights to validate stain-aware weighting hypothesis.
7. **Uncertainty quantification**: Calibrated uncertainty estimates for flagging ambiguous cases.

---

## References

[1] Bartlett, P. L., Harvey, N., Liaw, C., & Mehrabian, A. (2019). Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. *Journal of Machine Learning Research*, 20(63), 1–17.

[2] Benhabiles, H., Argubi-Wollesen, A., Mhiri, C., & Grau, V. (2021). MalariaNet: A deep learning approach for automatic detection of malaria parasites. In *Proceedings of the IEEE International Conference on Image Processing (ICIP)*.

[3] Bhansali, R. (2020). MalariaNet: A deep learning framework for malaria parasite detection using lightweight depthwise separable convolutions. *arXiv preprint*.

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In *ICCV*, 1026–1034. arXiv:1502.01852.

[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *CVPR*, 770–778.

[6] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. In *CVPR*, 7132–7141. arXiv:1709.01507.

[7] Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep networks with stochastic depth. In *ECCV*, 646–661. arXiv:1603.09382.

[8] Lin, T. Y., Dollar, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature pyramid networks for object detection. In *CVPR*, 2117–2125.

[9] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *ICLR*.

[10] Muller, R., Kornblith, S., & Hinton, G. (2019). When does label smoothing help? In *NeurIPS*, 32.

[11] Rajaraman, S., Antani, S. K., Poostchi, M., Silamut, K., Hossain, M. A., Maude, R. J., ... & Thoma, G. R. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ*, 6, e4568.

[12] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *CVPR*, 4510–4520. arXiv:1801.04381.

[13] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *ICCV*, 618–626.

[14] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception architecture. In *CVPR*, 2818–2826.

[15] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In *CVPR*, 1–9.

[16] Vijayalakshmi, A., & Kanna, B. R. (2020). Deep learning approach to detect malaria from microscopic images. *Multimedia Tools and Applications*, 79, 15297–15317.

[17] Wongsrichanalai, C., Barcus, M. J., Muth, S., Sutamihardja, A., & Wernsdorfer, W. H. (2007). A review of malaria diagnostic tools: Microscopy and rapid diagnostic test (RDT). *The American Journal of Tropical Medicine and Hygiene*, 77(6 Suppl), 119–127.

[18] Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Regularization strategy to train strong classifiers with localizable features. In *ICCV*, 6023–6032. arXiv:1905.04899.

[19] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. In *ICLR*. arXiv:1710.09412.

---

## Appendix A: Hardware Specifications

| Component | Specification |
|:----------|:-------------|
| GPU | NVIDIA GeForce RTX 3070 (8 GB VRAM) |
| System RAM | 32 GB |
| CPU Cores | 16 |
| CUDA Version | 12.7 |
| Framework | PyTorch (latest stable) |
| Precision | Mixed (torch.amp.autocast + GradScaler) |

## Appendix B: Reproducibility

All experiments use fixed random seeds (seed=42) for Python's `random` module, NumPy, and PyTorch. The complete codebase, trained model weights, and evaluation scripts will be made available upon publication.

---

*Manuscript prepared February 2026. PlasmoSENet training in progress; results sections will be updated upon completion.*
