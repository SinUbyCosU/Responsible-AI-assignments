"""
Q2: Does the Background Matter?
Responsible AI Assignment - Question 2

This script investigates whether binary image classifiers (cats vs. dogs)
learn genuine animal features or exploit a spurious background color correlation.

Setup:
  - Source data : CIFAR-10 cats (class 3) and dogs (class 5)
  - Compositing : each 32x32 animal is resized to 40x40 and placed at the
                  center of a 64x64 solid-color canvas, giving a clear 12px
                  background border on every side - no segmentation needed.
  - Training bias: 90% of cats get GREEN backgrounds, 90% of dogs get BLUE
                   backgrounds. The remaining 10% get the opposite color.

Three test sets are built from the same CIFAR-10 test split:
  IID            - same 90/10 correlation as training (in-distribution)
  Balanced       - background assigned randomly, 50/50 per class
  Counterfactual - correlation fully reversed (90% cats=blue, 90% dogs=green)

Two models are trained on the biased training set:
  CNN      - 3-block convolutional network trained from scratch
  ResNet18 - pretrained on ImageNet, then fine-tuned on our biased data

Metrics reported per model x test set:
  - Overall accuracy
  - Per-group accuracy: (background color) x (true class)
    e.g. green-cat, blue-cat, green-dog, blue-dog
  - Confusion matrix

Grad-CAM visualizations are saved as PNG files. If the heatmap lights up the
background border rather than the animal body, the model has learned a color
shortcut rather than genuine visual features.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use("Agg")   # render to file, no display needed
import matplotlib.pyplot as plt
from PIL import Image


# -- Config -------------------------------------------------------------------

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BIAS_STRENGTH = 0.9         # fraction of samples that follow the color rule
EPOCHS_CNN    = 12
EPOCHS_RESNET = 6           # fewer because pretrained weights give a head start
BATCH_SIZE    = 64
TOTAL_SIZE    = 64          # final square canvas size in pixels
ANIMAL_SIZE   = 40          # animal is resized to this before centering
DATA_DIR      = "./cifar10_data"
GRADCAM_DIR   = "./q2_gradcam"
GRADCAM_N     = 4           # correct + wrong examples to visualize per model/test-set

os.makedirs(GRADCAM_DIR, exist_ok=True)

CIFAR_CAT = 3   # CIFAR-10 label index for cat
CIFAR_DOG = 5   # CIFAR-10 label index for dog

# Background colors in uint8 RGB
GREEN = np.array([34, 139,  34], dtype=np.uint8)   # forest green -> cats (class 0)
BLUE  = np.array([30,  80, 200], dtype=np.uint8)   # medium blue  -> dogs (class 1)

print(f"Device: {DEVICE}")


# -- Dataset helpers ----------------------------------------------------------

def assign_bg(class_label, mode, bias_strength):
    """
    Picks a background color for one sample.

    class_label  : 0 = cat, 1 = dog
    mode         : 'biased' | 'balanced' | 'counterfactual'
    bias_strength: probability of following the rule (biased/counterfactual only)

    Biased:         cat -> GREEN, dog -> BLUE   (with prob = bias_strength)
    Balanced:       50/50 random, class-independent
    Counterfactual: cat -> BLUE, dog -> GREEN   (reversed, with prob = bias_strength)
    """
    if mode == "balanced":
        return GREEN.copy() if random.random() < 0.5 else BLUE.copy()

    if mode == "biased":
        correct, wrong = (GREEN, BLUE) if class_label == 0 else (BLUE,  GREEN)
    else:   # counterfactual: swap the rule
        correct, wrong = (BLUE,  GREEN) if class_label == 0 else (GREEN, BLUE)

    return correct.copy() if random.random() < bias_strength else wrong.copy()


def place_on_background(animal_np, bg_color):
    """
    Centers an (ANIMAL_SIZE, ANIMAL_SIZE, 3) uint8 animal image on a
    (TOTAL_SIZE, TOTAL_SIZE, 3) solid-color canvas.

    For our defaults (64x64 canvas, 40x40 animal) this gives a 12-pixel
    solid-color border on every side, which is (64^2 - 40^2) / 64^2 = 61%
    of the total image area. That is more than enough signal for a model
    to pick up on background color without looking at the animal at all.
    """
    canvas = np.full((TOTAL_SIZE, TOTAL_SIZE, 3), bg_color, dtype=np.uint8)
    offset = (TOTAL_SIZE - ANIMAL_SIZE) // 2   # = 12px each side
    canvas[offset : offset + ANIMAL_SIZE,
           offset : offset + ANIMAL_SIZE] = animal_np
    return canvas


class CatsDogsDataset(Dataset):
    """
    Filters CIFAR-10 down to cats and dogs, then composites each image onto
    a solid-color background according to the specified mode.

    Backgrounds are pre-assigned once in __init__ so:
      (a) the same random seed always gives the same dataset, and
      (b) we can look up bg_names[i] in evaluation to compute per-group accuracy.
    """

    def __init__(self, cifar_dataset, mode="biased", bias_strength=0.9):
        self.samples   = []   # list of (animal_np, binary_label)
        self.bg_colors = []   # pre-assigned background color per sample
        self.bg_names  = []   # 'green' or 'blue' for per-group analysis

        to_pil = transforms.ToPILImage()
        resize = transforms.Resize((ANIMAL_SIZE, ANIMAL_SIZE))

        for img_tensor, label in cifar_dataset:
            if label not in (CIFAR_CAT, CIFAR_DOG):
                continue

            binary_label = 0 if label == CIFAR_CAT else 1

            # img_tensor: (C, H, W) float [0,1] -- convert to resized uint8 HWC
            img_np = np.array(resize(to_pil(img_tensor)))   # (40, 40, 3) uint8

            bg = assign_bg(binary_label, mode, bias_strength)

            self.samples.append((img_np, binary_label))
            self.bg_colors.append(bg)
            self.bg_names.append("green" if np.array_equal(bg, GREEN) else "blue")

        # Normalization matching ImageNet statistics.
        # Using ImageNet stats is appropriate because ResNet18 was pretrained on
        # ImageNet. Applying the same normalization keeps its pretrained features
        # in their expected input range.
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        animal_np, label = self.samples[idx]
        composite = place_on_background(animal_np, self.bg_colors[idx])  # (64,64,3) uint8
        img = self.normalize(self.to_tensor(composite))   # (3,64,64) float, normalized
        return img, label


# -- Models -------------------------------------------------------------------

class CNN(nn.Module):
    """
    3-block convolutional network for 64x64 binary classification.
    BatchNorm after each conv stabilizes training on the small dataset.

    Feature map resolution:
      Input:  (3,  64, 64)
      block1: (32, 32, 32)  after MaxPool
      block2: (64, 16, 16)  after MaxPool
      block3: (128, 8,  8)  after MaxPool
      -> Flatten -> 8192 -> FC(256) -> Dropout(0.3) -> FC(2)
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
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        # Grad-CAM target: last conv layer (before BN/ReLU/Pool in block3)
        # Its output shape is (1, 128, 16, 16) -- spatial enough for a good heatmap
        self.target_layer = self.block3[0]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def get_resnet18():
    """
    ResNet18 pretrained on ImageNet, fine-tuned for binary (cat/dog) classification.

    We replace only the final FC layer (1000 classes -> 2 classes) and
    fine-tune all layers. This lets us test whether strong pretrained visual
    priors can resist the background color bias in our training data.

    The Grad-CAM target is the last conv layer in the last residual block.
    For 64x64 input, layer4 outputs a 2x2 spatial feature map.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.target_layer = model.layer4[-1].conv2   # (1, 512, 2, 2) for 64x64 input
    return model


# -- Training -----------------------------------------------------------------

def train(model, loader, epochs, lr=1e-3):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total   += len(labels)
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}  acc={correct/total:.1%}")


# -- Evaluation ---------------------------------------------------------------

def evaluate(model, dataset, loader):
    """
    Evaluates model on a test set and returns:
      overall_acc : float - fraction of correct predictions
      group_acc   : dict  - accuracy per (bg_color, true_class) combo
      cm          : 2x2 numpy array - confusion matrix [true x predicted]
      all_preds   : list of int - predicted labels in dataset order

    Since shuffle=False, dataset index == position in prediction list,
    so we can directly index dataset.bg_names[i] for per-group analysis.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Overall accuracy
    overall_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    # Per-group accuracy: key format is '{bg_color}-{true_class_name}'
    # This lets us see, for example, whether green-cats are classified correctly
    # vs blue-cats (which have the "wrong" background for their class).
    group_counts = {}   # key -> [n_correct, n_total]
    class_names  = ["cat", "dog"]

    for i, (pred, true_label) in enumerate(zip(all_preds, all_labels)):
        key = f"{dataset.bg_names[i]}-{class_names[true_label]}"
        if key not in group_counts:
            group_counts[key] = [0, 0]
        group_counts[key][0] += int(pred == true_label)
        group_counts[key][1] += 1

    group_acc = {k: v[0] / v[1] for k, v in group_counts.items()}

    # Confusion matrix: rows = true class, cols = predicted class
    cm = np.zeros((2, 2), dtype=int)
    for pred, true_label in zip(all_preds, all_labels):
        cm[true_label][pred] += 1

    return overall_acc, group_acc, cm, all_preds


def print_report(model_name, test_set_name, overall_acc, group_acc, cm):
    """Prints a formatted evaluation block for one model/test-set combination."""
    print(f"\n  [{model_name}] on [{test_set_name}]")
    print(f"  Overall accuracy: {overall_acc:.1%}")
    print(f"  Per-group breakdown:")
    for key in ["green-cat", "blue-cat", "green-dog", "blue-dog"]:
        acc = group_acc.get(key, float("nan"))
        print(f"    {key:<12}: {acc:.1%}")
    print(f"  Confusion matrix (rows=true, cols=predicted):")
    print(f"            Pred:cat  Pred:dog")
    print(f"  True:cat    {cm[0][0]:5d}     {cm[0][1]:5d}")
    print(f"  True:dog    {cm[1][0]:5d}     {cm[1][1]:5d}")


# -- Grad-CAM -----------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Reference: Selvaraju et al. 2017 (https://arxiv.org/abs/1610.02391)

    Algorithm:
      1. Forward pass: a hook on the target conv layer saves its output
         feature map A (shape: 1 x C x H x W).
      2. Backward pass: a tensor hook captures the gradient dL/dA.
      3. Global-average-pool the gradients over spatial dims -> weights alpha
         (shape: 1 x C x 1 x 1).
      4. Weighted sum of feature map channels: L = sum_c(alpha_c * A_c).
      5. ReLU(L) removes negative contributions, normalize to [0, 1].

    Interpretation:
      - Bright regions = where the model focused to make its decision.
      - If bright regions = background border -> color shortcut learner.
      - If bright regions = animal body -> genuine feature learner.
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None

        def save_activation(module, inp, output):
            # Save the feature map; also attach a hook on the tensor itself
            # so we capture the gradient when backward() is called.
            self.activations = output
            if output.requires_grad:
                output.register_hook(lambda grad: setattr(self, "gradients", grad))

        target_layer.register_forward_hook(save_activation)

    def generate(self, img_tensor):
        """
        img_tensor : (C, H, W) normalized tensor for a single image
        Returns    : (cam, pred_class)
                     cam is a (H, W) float array in [0, 1]
        """
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(DEVICE)   # (1, C, H, W)

        self.model.zero_grad()
        out = self.model(x)                       # forward; hook fires -> saves activations
        pred_class = out.argmax(dim=1).item()
        out[0, pred_class].backward()             # backward; tensor hook -> saves gradients

        # alpha: global average pool of gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)

        # Weighted sum of activation channels -> spatial attention map (H_feat, W_feat)
        cam = (weights * self.activations).sum(dim=1).squeeze()    # (H_feat, W_feat)
        cam = torch.relu(cam).detach().cpu().numpy()

        # Normalize to [0, 1] so we can apply a colormap
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_class


def save_gradcam_grid(model, gradcam, dataset, test_set_name, model_name):
    """
    Saves a PNG grid of Grad-CAM visualizations.
    Collects up to GRADCAM_N correctly-classified and GRADCAM_N misclassified
    examples, then for each shows: original | heatmap | overlay.

    Misclassified examples are especially informative: if the model predicts
    the wrong class AND the Grad-CAM shows the background region, that is
    direct evidence of shortcut reliance.
    """
    model.eval()
    class_names  = ["cat", "dog"]
    correct_idxs = []
    wrong_idxs   = []

    # Iterate through dataset sequentially to find correct/wrong examples
    with torch.no_grad():
        for idx in range(len(dataset)):
            if len(correct_idxs) >= GRADCAM_N and len(wrong_idxs) >= GRADCAM_N:
                break
            img, label = dataset[idx]
            pred = model(img.unsqueeze(0).to(DEVICE)).argmax(1).item()
            if pred == label and len(correct_idxs) < GRADCAM_N:
                correct_idxs.append(idx)
            elif pred != label and len(wrong_idxs) < GRADCAM_N:
                wrong_idxs.append(idx)

    all_examples = [(i, "CORRECT") for i in correct_idxs] + \
                   [(i, "WRONG")   for i in wrong_idxs]

    if not all_examples:
        print(f"    No examples to visualize for {model_name} / {test_set_name}")
        return

    # Inverse normalization: convert normalized tensor back to displayable [0,1] image
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std =[1/0.229,      1/0.224,      1/0.225]
    )

    nrows = len(all_examples)
    fig, axes = plt.subplots(nrows, 3, figsize=(10, 3.2 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, (idx, status) in enumerate(all_examples):
        img_tensor, true_label = dataset[idx]

        # Generate Grad-CAM (requires gradient computation, done outside no_grad)
        cam, pred_class = gradcam.generate(img_tensor)

        # Denormalize original image for display
        orig = inv_norm(img_tensor).permute(1, 2, 0).numpy()
        orig = np.clip(orig, 0, 1)

        # Upsample cam from feature-map resolution to full image resolution
        cam_pil    = Image.fromarray((cam * 255).astype(np.uint8))
        cam_pil    = cam_pil.resize((TOTAL_SIZE, TOTAL_SIZE), Image.BILINEAR)
        cam_np     = np.array(cam_pil) / 255.0              # (64, 64) in [0, 1]
        cam_colored = plt.cm.jet(cam_np)[:, :, :3]          # apply jet colormap -> (64,64,3)

        # Blend original + colored heatmap for the overlay
        overlay = 0.55 * orig + 0.45 * cam_colored

        bg    = dataset.bg_names[idx]
        label = f"{status} | true={class_names[true_label]} pred={class_names[pred_class]} | bg={bg}"

        axes[row, 0].imshow(orig);                                  axes[row, 0].set_title(f"Original\n{label}", fontsize=8)
        axes[row, 1].imshow(cam_np, cmap="jet", vmin=0, vmax=1);   axes[row, 1].set_title("Grad-CAM", fontsize=8)
        axes[row, 2].imshow(overlay);                               axes[row, 2].set_title("Overlay", fontsize=8)
        for col in range(3):
            axes[row, col].axis("off")

    plt.suptitle(f"Grad-CAM | {model_name} | Test: {test_set_name}", fontsize=11, y=1.01)
    plt.tight_layout()

    safe = test_set_name.replace(" ", "_")
    fname = os.path.join(GRADCAM_DIR, f"{model_name}_{safe}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"    Saved: {fname}")


# -- Interpretation -----------------------------------------------------------

def interpret(model_name, results):
    """
    Prints a short interpretation paragraph based on accuracy patterns.

    Key signals:
    - IID >> Balanced or Counterfactual  -> model relies on bg color
    - green-cat >> blue-cat accuracy     -> model associates green with correct
    - Grad-CAM heatmap on border         -> confirms bg-color shortcut
    """
    iid_acc = results["IID"][0]
    bal_acc = results["Balanced"][0]
    cf_acc  = results["Counterfactual"][0]
    drop    = iid_acc - cf_acc

    print(f"\n  {model_name}:")
    print(f"    IID={iid_acc:.1%}  Balanced={bal_acc:.1%}  Counterfactual={cf_acc:.1%}")
    print(f"    Accuracy drop (IID -> Counterfactual): {drop:.1%}")

    # Check per-group gap: if green-cat >> blue-cat, model uses background
    gc = results["IID"][1].get("green-cat", 0)
    bc = results["IID"][1].get("blue-cat",  0)
    group_gap = gc - bc

    if drop > 0.30:
        verdict = "HEAVY background reliance (shortcut learner). " \
                  "Model likely ignores the animal and uses border color."
    elif drop > 0.10:
        verdict = "PARTIAL background reliance. " \
                  "Model uses both color cue and some animal features."
    else:
        verdict = "ROBUST to background. " \
                  "Model appears to have learned genuine animal features."

    if group_gap > 0.15:
        verdict += f" Per-group gap ({gc:.0%} vs {bc:.0%} on IID) confirms bg-color usage."

    print(f"    Verdict: {verdict}")


# -- Main ---------------------------------------------------------------------

def main():
    random.seed(42)
    torch.manual_seed(42)

    # Load CIFAR-10 - just ToTensor (no normalization yet; Dataset handles that)
    raw_t = transforms.ToTensor()
    cifar_train = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=raw_t)
    cifar_test  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=raw_t)

    # Build all datasets
    # Training: biased (green=cat, blue=dog, 90/10)
    # Test IID: same bias distribution as training
    # Test Balanced: background randomly assigned, no class correlation
    # Test Counterfactual: reversed rule (green=dog, blue=cat, 90/10)
    print("\nBuilding datasets (filtering cats/dogs, compositing backgrounds)...")
    train_ds = CatsDogsDataset(cifar_train, mode="biased",          bias_strength=BIAS_STRENGTH)
    test_iid = CatsDogsDataset(cifar_test,  mode="biased",          bias_strength=BIAS_STRENGTH)
    test_bal = CatsDogsDataset(cifar_test,  mode="balanced",        bias_strength=0.5)
    test_cf  = CatsDogsDataset(cifar_test,  mode="counterfactual",  bias_strength=BIAS_STRENGTH)

    print(f"  Train: {len(train_ds)} samples | IID test: {len(test_iid)} | "
          f"Balanced: {len(test_bal)} | Counterfactual: {len(test_cf)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    iid_loader   = DataLoader(test_iid, batch_size=BATCH_SIZE, shuffle=False)
    bal_loader   = DataLoader(test_bal, batch_size=BATCH_SIZE, shuffle=False)
    cf_loader    = DataLoader(test_cf,  batch_size=BATCH_SIZE, shuffle=False)

    # Train both models on the biased training set
    print("\n" + "="*60)
    print("Training CNN (from scratch) on biased training data...")
    cnn = CNN()
    train(cnn, train_loader, EPOCHS_CNN, lr=1e-3)

    print(f"\nFine-tuning ResNet18 (pretrained ImageNet) on biased training data...")
    resnet = get_resnet18()
    train(resnet, train_loader, EPOCHS_RESNET, lr=5e-4)

    # Evaluate all combinations
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    test_configs = [
        ("IID",            test_iid, iid_loader),
        ("Balanced",       test_bal, bal_loader),
        ("Counterfactual", test_cf,  cf_loader),
    ]

    all_results = {}

    for model_name, model in [("CNN", cnn), ("ResNet18", resnet)]:
        all_results[model_name] = {}
        gradcam = GradCAM(model, model.target_layer)

        for test_name, test_ds, test_loader in test_configs:
            overall, group_acc, cm, preds = evaluate(model, test_ds, test_loader)
            print_report(model_name, test_name, overall, group_acc, cm)
            all_results[model_name][test_name] = (overall, group_acc, cm)

            print(f"  Generating Grad-CAM visualizations...")
            save_gradcam_grid(model, gradcam, test_ds, test_name, model_name)

    # Final summary table
    print("\n" + "="*75)
    print("FINAL SUMMARY TABLE")
    print("="*75)
    print(f"{'Model':<12} {'Test Set':<17} {'Acc':>6}  {'gr-cat':>7} {'bl-cat':>7} {'gr-dog':>7} {'bl-dog':>7}")
    print("-"*68)
    for model_name in ["CNN", "ResNet18"]:
        for test_name, _, _ in test_configs:
            overall, group_acc, _ = all_results[model_name][test_name]
            print(
                f"{model_name:<12} {test_name:<17} {overall:>5.1%}"
                f"  {group_acc.get('green-cat',0):>6.1%}"
                f" {group_acc.get('blue-cat',0):>6.1%}"
                f" {group_acc.get('green-dog',0):>6.1%}"
                f" {group_acc.get('blue-dog',0):>6.1%}"
            )

    # Interpretation
    print("\n" + "="*75)
    print("INTERPRETATION")
    print("="*75)
    for model_name in ["CNN", "ResNet18"]:
        interpret(model_name, all_results[model_name])

    print(f"\nGrad-CAM images saved to: {os.path.abspath(GRADCAM_DIR)}/")
    print("Done!")


if __name__ == "__main__":
    main()
