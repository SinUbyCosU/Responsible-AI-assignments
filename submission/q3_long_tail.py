"""
Q3: The Long Tail Problem
Responsible AI Assignment - Question 3

Goal:
  Demonstrate that standard cross-entropy training on a long-tailed dataset
  causes models to collapse on rare ("tail") classes, and compare three
  remedies that explicitly address this.

Dataset:
  CIFAR-10, subsampled to 7 classes with this distribution:
    airplane (cls 0):   5000 samples   <- HEAD
    automobile (cls 1): 2500 samples
    bird (cls 2):       1250 samples
    cat (cls 3):         500 samples
    deer (cls 4):        250 samples
    dog (cls 5):          50 samples
    frog (cls 6):         25 samples   <- TAIL
  Imbalance ratio: 5000/25 = 200:1

  Note: The assignment's Table 1 specifies [10000, 5000, ..., 50]. CIFAR-10
  only has 5000 samples per class, so all counts are halved. The imbalance
  ratio is identical (200:1), so the long-tail effect is the same.

  Test set: 1000 samples per class x 7 classes = 7000 total, perfectly balanced.

Models:
  CNN     - 3-block convolutional network (same architecture across all runs)
  TinyViT - small Vision Transformer, patch_size=4 on 32x32 images

Training runs (IDENTICAL architecture, optimizer, epochs - only loss/sampler differs):
  1. Baseline       - standard cross-entropy, no class handling
  2. Weighted CE    - cross-entropy with per-class weights = total / (K * n_k)
  3. Oversampling   - WeightedRandomSampler: rare classes sampled more per epoch
  4. Focal Loss     - (1 - p_t)^gamma modulating factor, gamma=2

Metrics:
  per_class_acc  : accuracy for each class individually
  balanced_acc   : mean of per-class recalls (not biased by class size)
  worst_class    : min per-class accuracy (the tail class floor)
  overall_acc    : standard accuracy (for reference; inflated by head classes)
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

# Config
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
EPOCHS      = 15       # same for ALL runs - essential for a fair comparison
BATCH_SIZE  = 128
LR          = 1e-3
DATA_DIR    = "./cifar10_data"

# CIFAR-10 indices for the 7 classes we keep
CIFAR_CLASSES = [0, 1, 2, 3, 4, 5, 6]
CLASS_NAMES   = ["airplane", "auto", "bird", "cat", "deer", "dog", "frog"]

# Training counts (halved from assignment Table 1 to fit CIFAR-10's 5000/class cap).
# The 200:1 head-to-tail imbalance ratio is identical to the original spec.
TRAIN_COUNTS = [5000, 2500, 1250, 500, 250, 50, 25]

# CIFAR-10 channel normalization stats (computed from the full training set)
NORMALIZE = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std =[0.2023, 0.1994, 0.2010]
)

print(f"Device: {DEVICE}")
print(f"Imbalance ratio: {TRAIN_COUNTS[0]}/{TRAIN_COUNTS[-1]} = {TRAIN_COUNTS[0]//TRAIN_COUNTS[-1]}:1")


# Dataset
def load_and_subsample(data_dir):
    """
    Loads CIFAR-10 and builds:
      - A long-tailed training set with exactly TRAIN_COUNTS[i] images per class
      - A balanced test set with all 1000 images per class

    We access CIFAR-10's internal .data and .targets numpy arrays directly
    instead of iterating the Dataset object (much faster for filtering).
    """
    cifar_train = datasets.CIFAR10(data_dir, train=True,  download=True)
    cifar_test  = datasets.CIFAR10(data_dir, train=False, download=True)

    train_imgs    = cifar_train.data            # (50000, 32, 32, 3) uint8
    train_targets = np.array(cifar_train.targets)
    test_imgs     = cifar_test.data             # (10000, 32, 32, 3) uint8
    test_targets  = np.array(cifar_test.targets)

    # Build imbalanced training set:
    # For each of the 7 classes, randomly pick exactly TRAIN_COUNTS[i] samples.
    # np.where gives us all indices for that class; we then sample without replacement.
    train_samples = []
    for new_label, (orig_label, n) in enumerate(zip(CIFAR_CLASSES, TRAIN_COUNTS)):
        all_idxs = np.where(train_targets == orig_label)[0]
        chosen   = np.random.choice(all_idxs, size=min(n, len(all_idxs)), replace=False)
        for i in chosen:
            train_samples.append((train_imgs[i], new_label))

    # Build balanced test set: keep ALL test images from each of the 7 classes.
    # CIFAR-10 test has exactly 1000 per class, so this gives 7000 balanced samples.
    test_samples = []
    for new_label, orig_label in enumerate(CIFAR_CLASSES):
        idxs = np.where(test_targets == orig_label)[0]
        for i in idxs:
            test_samples.append((test_imgs[i], new_label))

    # Shuffle so class groups aren't contiguous in training
    random.shuffle(train_samples)

    print(f"  Train: {len(train_samples)} samples (imbalanced)")
    print(f"  Test : {len(test_samples)} samples (balanced, {len(test_samples)//NUM_CLASSES}/class)")
    return train_samples, test_samples


class CIFARSubset(Dataset):
    """
    Minimal Dataset wrapper for our subsampled CIFAR-10.
    Stores images as uint8 numpy arrays (memory-efficient) and converts
    to normalized float tensors on-the-fly in __getitem__.
    """

    def __init__(self, samples):
        self.samples = samples
        self.labels  = [lbl for _, lbl in samples]   # kept separately for fast access
        self.transform = transforms.Compose([
            transforms.ToTensor(),   # (H,W,C) uint8 -> (C,H,W) float in [0,1]
            NORMALIZE,
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, label = self.samples[idx]
        return self.transform(img_np), label


def make_standard_loader(dataset, shuffle=True):
    """Plain DataLoader - classes appear in proportion to their training set size."""
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def make_oversampled_loader(dataset):
    """
    DataLoader with WeightedRandomSampler for oversampling intervention.

    Each sample is assigned weight = 1 / n_k where n_k is the training count
    of its class. When sampling with replacement, this makes every class
    equally likely to appear in a batch, regardless of how rare it is.

    The number of samples drawn per epoch equals len(dataset) (same as baseline),
    so training time and number of gradient steps are comparable.

    Key difference from weighted loss: here we change WHICH samples are seen,
    not HOW MUCH each sample contributes to the loss.
    """
    sample_weights = [1.0 / TRAIN_COUNTS[lbl] for lbl in dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True   # required: rare classes must be reused to match head frequency
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)


# Models

class CNN(nn.Module):
    """
    3-block CNN for 32x32 CIFAR images.

    Architecture:
      Input  (3, 32, 32)
      Block1 (32, 16, 16) after MaxPool
      Block2 (64,  8,  8) after MaxPool
      Block3 (128, 4,  4) after MaxPool
      Flatten -> Linear(2048, 256) -> Dropout(0.3) -> Linear(256, 7)
    """

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,  32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))


class TinyViT(nn.Module):
    """
    Small Vision Transformer for 32x32 CIFAR images.

    patch_size=4 divides 32x32 into 8x8 = 64 spatial patches.
    Each patch is embedded to embed_dim=64 via a single Conv2d.
    4 transformer layers with 4 attention heads.
    Classification uses the [CLS] token output.
    """

    def __init__(self, patch_size=4, embed_dim=64, num_heads=4, num_layers=4):
        super().__init__()
        num_patches = (32 // patch_size) ** 2   # 64 patches for patch_size=4

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True  # pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x):
        B = x.shape[0]
        x   = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, 64, 64)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.head(self.transformer(x)[:, 0])             # CLS token -> classes


# Loss Functions

def get_class_weights():
    """
    Per-class weights for weighted cross-entropy.

    Formula: weight_k = total_samples / (K * n_k)

    This ensures that the expected loss contribution from each class is equal,
    counteracting the imbalance by up-weighting rare classes.
    """
    total = sum(TRAIN_COUNTS)
    return torch.tensor(
        [total / (NUM_CLASSES * n) for n in TRAIN_COUNTS],
        dtype=torch.float,
        device=DEVICE
    )


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017, arXiv:1708.02002).

    Modification of cross-entropy that adds a modulating factor:
      FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    When p_t is high (model is confident and correct), (1-p_t)^gamma -> 0,
    so the loss contribution from easy examples is down-weighted.
    When p_t is low (model is wrong or uncertain), the factor approaches 1,
    preserving the standard CE loss for hard examples.

    gamma=0 : reduces to standard cross-entropy
    gamma=2 : standard choice from the original paper
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t     = torch.exp(-ce_loss)                     # prob of correct class
        return ((1 - p_t) ** self.gamma * ce_loss).mean()


# Training

def train(model, loader, epochs, criterion):
    """
    Trains model for a fixed number of epochs.
    The criterion (loss function) is the only thing that varies between runs.
    Everything else - optimizer, LR, epochs - stays identical for fair comparison.
    """
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)
        print(f"  Epoch {epoch+1:2d}/{epochs}  loss={total_loss/len(loader):.4f}  train_acc={correct/total:.1%}")


# Evaluation

def evaluate(model, loader):
    """
    Computes four metrics on the balanced test set.

    per_class_acc (list of float):
      Accuracy for each class independently.
      The key diagnostic: under the baseline, tail class accuracy will be near 0%
      because the model sees so few examples of those classes during training.

    balanced_acc (float):
      = mean(per_class_acc)
      Unlike overall accuracy, this treats all 7 classes equally.
      A model that predicts only head classes would score ~14% (1/7) here.

    worst_class_acc (float):
      = min(per_class_acc)
      The single most important metric for the long-tail problem.
      Shows the model's worst-case failure on the rarest class.

    overall_acc (float):
      Standard accuracy, included for reference.
      Note: this can look deceptively good under the baseline because the
      test set is balanced, but the model is still failing on tail classes.
    """
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            for pred, label in zip(preds.tolist(), labels.tolist()):
                class_total[label]   += 1
                class_correct[label] += int(pred == label)

    per_class_acc   = [class_correct[i] / class_total[i] for i in range(NUM_CLASSES)]
    balanced_acc    = sum(per_class_acc) / NUM_CLASSES
    worst_class_acc = min(per_class_acc)
    overall_acc     = sum(class_correct) / sum(class_total)

    return per_class_acc, balanced_acc, worst_class_acc, overall_acc


# Results Table

def print_results_table(all_results):
    """
    Prints two summary tables: one for CNN, one for ViT.
    Each table shows per-class accuracy for all 4 runs side by side,
    sorted head -> tail, so the long-tail effect and improvement are visible.
    """
    run_names = list(all_results.keys())

    for arch in ["CNN", "ViT"]:
        print(f"\n{'='*72}")
        print(f"  {arch} — Per-Class Accuracy (sorted head to tail)")
        print(f"{'='*72}")
        header = f"  {'Class':<12} {'N_train':>7}  "
        for r in run_names:
            header += f"  {r:<15}"
        print(header)
        print(f"  {'-'*68}")

        for cls_idx in range(NUM_CLASSES):
            row = f"  {CLASS_NAMES[cls_idx]:<12} {TRAIN_COUNTS[cls_idx]:>7}  "
            for r in run_names:
                acc = all_results[r][arch][0][cls_idx]
                row += f"  {acc:>14.1%}"
            print(row)

        print(f"  {'-'*68}")
        for label, metric_idx in [("Balanced Acc", 1), ("Worst Class", 2), ("Overall Acc", 3)]:
            row = f"  {label:<12} {'---':>7}  "
            for r in run_names:
                val = all_results[r][arch][metric_idx]
                row += f"  {val:>14.1%}"
            print(row)

        print(f"{'='*72}")


def print_combined_summary(all_results):
    """Prints a compact summary of balanced_acc and worst_class across all runs x models."""
    run_names = list(all_results.keys())
    print(f"\n{'='*70}")
    print("  COMBINED SUMMARY (all runs x models)")
    print(f"{'='*70}")
    print(f"  {'Run':<18}  {'Model':<6}  {'Balanced Acc':>13}  {'Worst Class':>12}  {'Overall Acc':>12}")
    print(f"  {'-'*65}")
    for r in run_names:
        for arch in ["CNN", "ViT"]:
            _, bal, worst, overall = all_results[r][arch]
            print(f"  {r:<18}  {arch:<6}  {bal:>12.1%}  {worst:>12.1%}  {overall:>12.1%}")
    print(f"{'='*70}")


# Main

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("\nLoading CIFAR-10 and building long-tailed training set...")
    train_samples, test_samples = load_and_subsample(DATA_DIR)

    train_ds = CIFARSubset(train_samples)
    test_ds  = CIFARSubset(test_samples)

    # Create DataLoaders
    standard_loader    = make_standard_loader(train_ds)
    oversampled_loader = make_oversampled_loader(train_ds)
    test_loader        = make_standard_loader(test_ds, shuffle=False)

    # Loss functions - these are the ONLY differences between runs
    ce_standard = nn.CrossEntropyLoss()
    ce_weighted = nn.CrossEntropyLoss(weight=get_class_weights())
    focal       = FocalLoss(gamma=2.0)

    # Run configurations: (name, training loader, loss criterion)
    runs = [
        ("Baseline (CE)",  standard_loader,    ce_standard),
        ("Weighted CE",    standard_loader,    ce_weighted),   # same data, different loss
        ("Oversampling",   oversampled_loader,  ce_standard),   # same loss, different sampling
        ("Focal Loss",     standard_loader,    focal),          # same data, different loss
    ]

    all_results = {}   # {run_name: {arch_name: (per_class_acc, bal, worst, overall)}}

    for run_name, loader, criterion in runs:
        print("\n" + "="*60)
        print(f"Run: {run_name}")
        all_results[run_name] = {}

        for arch_name, ModelClass in [("CNN", CNN), ("ViT", TinyViT)]:
            print(f"\n  Training {arch_name}...")
            model = ModelClass()
            train(model, loader, EPOCHS, criterion)

            print(f"  Evaluating {arch_name}...")
            metrics = evaluate(model, test_loader)
            all_results[run_name][arch_name] = metrics

            per_class, balanced, worst, overall = metrics
            print(f"  Balanced={balanced:.1%}  Worst-class={worst:.1%}  Overall={overall:.1%}")

    # Print full tables
    print_results_table(all_results)
    print_combined_summary(all_results)

    # Print a plain text summary for the report
    print("\n" + "="*60)
    print("KEY FINDINGS TO REPORT:")
    print("="*60)
    base_cnn = all_results["Baseline (CE)"]["CNN"]
    base_vit = all_results["Baseline (CE)"]["ViT"]
    print(f"\nBaseline worst-class accuracy:")
    print(f"  CNN: {base_cnn[2]:.1%}  |  ViT: {base_vit[2]:.1%}")
    print(f"  (If this is near 0%, the model has completely collapsed on the tail)")

    for run_name in ["Weighted CE", "Oversampling", "Focal Loss"]:
        cnn = all_results[run_name]["CNN"]
        vit = all_results[run_name]["ViT"]
        base_cnn_worst = base_cnn[2]
        print(f"\n{run_name} improvement (worst-class):")
        print(f"  CNN: {base_cnn[2]:.1%} -> {cnn[2]:.1%}  ({cnn[2]-base_cnn[2]:+.1%})")
        print(f"  ViT: {base_vit[2]:.1%} -> {vit[2]:.1%}  ({vit[2]-base_vit[2]:+.1%})")

    print("\nDone!")


if __name__ == "__main__":
    main()
