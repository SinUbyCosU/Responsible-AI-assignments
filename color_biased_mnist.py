"""
Color-Biased MNIST Experiment
Responsible AI Assignment

What this does:
  - Creates a colored MNIST where digits 0-4 are RED and 5-9 are BLUE (biased)
  - bias_strength=0.9 means 90% of images follow the rule, 10% get a random color
  - Trains a CNN and a small ViT on the biased colored data
  - Also trains both on plain grayscale MNIST for comparison
  - Tests each model under 3 conditions: same bias / random color / reversed color
  - Prints a final results table
  - Runs a shape-vs-color diagnostic to check if models use color as a shortcut
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import random

# -- Config -------------------------------------------------------------------
BIAS_STRENGTH = 0.9      # fraction of images that follow the color rule
EPOCHS_CNN    = 5
EPOCHS_VIT    = 8
BATCH_SIZE    = 128
DATA_DIR      = "./mnist_data"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Color assignments: foreground of digit pixels
RED  = np.array([1.0, 0.0, 0.0])   # for digits 0-4
BLUE = np.array([0.0, 0.0, 1.0])   # for digits 5-9

print(f"Device: {DEVICE}")


# -- Dataset ------------------------------------------------------------------

def pick_color(label, mode, bias_strength):
    """
    Returns an RGB color for the digit foreground.

    mode='biased'   -> 0-4 get RED, 5-9 get BLUE (with bias_strength probability)
    mode='reversed' -> 0-4 get BLUE, 5-9 get RED (with bias_strength probability)
    mode='random'   -> completely random RGB color, ignores label
    """
    if mode == "random":
        return np.array([random.random(), random.random(), random.random()])

    if mode == "biased":
        correct, wrong = (RED, BLUE) if label < 5 else (BLUE, RED)
    else:  # reversed
        correct, wrong = (BLUE, RED) if label < 5 else (RED, BLUE)

    return correct if random.random() < bias_strength else wrong


class ColoredMNIST(Dataset):
    """
    Wraps raw MNIST and colorizes digit pixels (foreground coloring).
    Colors are pre-assigned once so the same image always gets the same color.
    """

    def __init__(self, base_dataset, mode="biased", bias_strength=0.9):
        self.base   = base_dataset
        # Pre-assign a color to every sample up front
        self.colors = [pick_color(label, mode, bias_strength)
                       for _, label in base_dataset]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]       # img shape: (1, 28, 28), range [0, 1]
        gray = img.squeeze().numpy()      # (28, 28)
        c = self.colors[idx]
        # Multiply grayscale intensity by the color -- background stays black
        rgb = np.stack([gray * c[0], gray * c[1], gray * c[2]], axis=0)
        return torch.FloatTensor(rgb), label


# -- Models -------------------------------------------------------------------

class CNN(nn.Module):
    """Simple 2-block CNN."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class TinyViT(nn.Module):
    """
    Small Vision Transformer for 28x28 images.
    patch_size=7 -> 4x4 = 16 patches per image.
    """

    def __init__(self, in_channels=3, patch_size=7, embed_dim=64, num_heads=4, num_layers=4):
        super().__init__()
        num_patches = (28 // patch_size) ** 2  # 16

        # Turn each patch into an embedding vector via a strided convolution
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True,  # pre-LN is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, 10)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])  # CLS token -> classification


# -- Training & Evaluation ----------------------------------------------------

def train(model, loader, epochs):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")


def get_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(dim=1) == labels).sum().item()
            total   += len(labels)
    return correct / total


# -- Shape vs Color Diagnostic ------------------------------------------------

def shape_vs_color_diagnostic(model_name, model, correct_color_loader, reversed_color_loader):
    """
    Tests the model on images with correct color vs reversed color.

    If accuracy drops a lot when the color is reversed, the model learned
    to use color as a shortcut rather than actually reading the digit shape.
    """
    acc_correct  = get_accuracy(model, correct_color_loader)
    acc_reversed = get_accuracy(model, reversed_color_loader)
    drop = acc_correct - acc_reversed

    print(f"\n{model_name}:")
    print(f"  Accuracy -- correct color : {acc_correct:.1%}")
    print(f"  Accuracy -- reversed color: {acc_reversed:.1%}")
    print(f"  Accuracy drop             : {drop:.1%}")

    if drop > 0.25:
        verdict = "Relies HEAVILY on color -> shortcut learner"
    elif drop > 0.08:
        verdict = "Uses both color AND shape"
    else:
        verdict = "Relies mostly on digit shape -> robust"
    print(f"  Verdict                   : {verdict}")

    return acc_correct, acc_reversed, drop


# -- Main ---------------------------------------------------------------------

def main():
    random.seed(42)
    torch.manual_seed(42)

    # Load raw MNIST
    to_tensor = transforms.ToTensor()
    train_raw = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=to_tensor)
    test_raw  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_tensor)

    # Build colored datasets
    print("\nBuilding colored datasets (assigns colors once, then caches them)...")
    train_biased  = ColoredMNIST(train_raw, mode="biased",   bias_strength=BIAS_STRENGTH)
    test_biased   = ColoredMNIST(test_raw,  mode="biased",   bias_strength=BIAS_STRENGTH)
    test_random   = ColoredMNIST(test_raw,  mode="random",   bias_strength=BIAS_STRENGTH)
    test_reversed = ColoredMNIST(test_raw,  mode="reversed", bias_strength=BIAS_STRENGTH)

    # DataLoaders
    train_colored_loader  = DataLoader(train_biased,  batch_size=BATCH_SIZE, shuffle=True)
    test_biased_loader    = DataLoader(test_biased,   batch_size=BATCH_SIZE)
    test_random_loader    = DataLoader(test_random,   batch_size=BATCH_SIZE)
    test_reversed_loader  = DataLoader(test_reversed, batch_size=BATCH_SIZE)

    train_gray_loader = DataLoader(train_raw, batch_size=BATCH_SIZE, shuffle=True)
    test_gray_loader  = DataLoader(test_raw,  batch_size=BATCH_SIZE)

    # Train all 4 models
    print("\n" + "=" * 60)
    print(f"Training CNN on COLOR-BIASED data (bias={BIAS_STRENGTH})...")
    cnn_color = CNN(in_channels=3)
    train(cnn_color, train_colored_loader, EPOCHS_CNN)

    print(f"\nTraining ViT on COLOR-BIASED data (bias={BIAS_STRENGTH})...")
    vit_color = TinyViT(in_channels=3)
    train(vit_color, train_colored_loader, EPOCHS_VIT)

    print("\nTraining CNN on GRAYSCALE data (no color)...")
    cnn_gray = CNN(in_channels=1)
    train(cnn_gray, train_gray_loader, EPOCHS_CNN)

    print("\nTraining ViT on GRAYSCALE data (no color)...")
    vit_gray = TinyViT(in_channels=1)
    train(vit_gray, train_gray_loader, EPOCHS_VIT)

    # Evaluate colored models under 3 test conditions
    print("\n" + "=" * 60)
    print("Evaluating...")

    results = {
        "CNN (color)": {
            "Same bias (0.9)": get_accuracy(cnn_color, test_biased_loader),
            "Random color":    get_accuracy(cnn_color, test_random_loader),
            "Reversed color":  get_accuracy(cnn_color, test_reversed_loader),
        },
        "ViT (color)": {
            "Same bias (0.9)": get_accuracy(vit_color, test_biased_loader),
            "Random color":    get_accuracy(vit_color, test_random_loader),
            "Reversed color":  get_accuracy(vit_color, test_reversed_loader),
        },
    }

    # Grayscale models don't see color at all, so all 3 conditions give the same
    # result. We test once on plain grayscale and repeat it across all columns.
    gray_cnn_acc = get_accuracy(cnn_gray, test_gray_loader)
    gray_vit_acc = get_accuracy(vit_gray, test_gray_loader)

    results["CNN (grayscale)"] = {c: gray_cnn_acc for c in ["Same bias (0.9)", "Random color", "Reversed color"]}
    results["ViT (grayscale)"] = {c: gray_vit_acc for c in ["Same bias (0.9)", "Random color", "Reversed color"]}

    # Print final comparison table
    print("\n" + "=" * 65)
    print("FINAL RESULTS TABLE")
    print("=" * 65)
    print(f"{'Model':<20} {'Same Bias':>12} {'Random Color':>14} {'Reversed':>12}")
    print("-" * 65)
    for model_name, accs in results.items():
        print(
            f"{model_name:<20}"
            f" {accs['Same bias (0.9)']:>11.1%}"
            f" {accs['Random color']:>13.1%}"
            f" {accs['Reversed color']:>11.1%}"
        )
    print()
    print("Note: Grayscale model accuracy is the same across all 3 columns")
    print("because color is invisible to them -- they only see digit shape.")

    # Shape vs Color Diagnostic
    print("\n" + "=" * 65)
    print("SHAPE VS COLOR DIAGNOSTIC")
    print("If a model relies on color instead of digit shape, its accuracy")
    print("will drop sharply when color labels are reversed.")
    print("=" * 65)

    shape_vs_color_diagnostic("CNN (color)", cnn_color, test_biased_loader, test_reversed_loader)
    shape_vs_color_diagnostic("ViT (color)", vit_color, test_biased_loader, test_reversed_loader)

    print("\nDone!")


if __name__ == "__main__":
    main()
