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

# Set the fraction of images that follow the color rule
BIAS_STRENGTH = 0.9      
# Set how many times we loop through the data for the CNN
EPOCHS_CNN    = 5
# Set how many times we loop through the data for the ViT
EPOCHS_VIT    = 8
# Set the batch size for training
BATCH_SIZE    = 128
# Directory where the MNIST data will be downloaded
DATA_DIR      = "./mnist_data"
# Use GPU if we have one, otherwise fallback to CPU
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Define the colors we'll use for the foreground of the digits
RED  = np.array([1.0, 0.0, 0.0])   # Red color for digits 0-4
BLUE = np.array([0.0, 0.0, 1.0])   # Blue color for digits 5-9

# Just print out what device we are running on
print(f"Device: {DEVICE}")

def pick_color(label, mode, bias_strength):
    """
    Returns an RGB color for the digit foreground.

    mode='biased'   -> 0-4 get RED, 5-9 get BLUE (with bias_strength probability)
    mode='reversed' -> 0-4 get BLUE, 5-9 get RED (with bias_strength probability)
    mode='random'   -> completely random RGB color, ignores label
    """
    # If the mode is random, just pick any random color
    if mode == "random":
        return np.array([random.random(), random.random(), random.random()])

    # If the mode is biased, 0-4 get red, otherwise blue
    if mode == "biased":
        correct, wrong = (RED, BLUE) if label < 5 else (BLUE, RED)
    # If the mode is reversed, 0-4 get blue, otherwise red
    else:  
        correct, wrong = (BLUE, RED) if label < 5 else (RED, BLUE)

    # Apply the bias strength. Usually we return the 'correct' color, but sometimes we inject the wrong one
    return correct if random.random() < bias_strength else wrong


class ColoredMNIST(Dataset):
    """
    Wraps raw MNIST and colorizes digit pixels (foreground coloring).
    Colors are pre-assigned once so the same image always gets the same color.
    """

    def __init__(self, base_dataset, mode="biased", bias_strength=0.9):
        # Save the original grayscale dataset
        self.base   = base_dataset
        # Pre-assign a color to every single sample right at the beginning
        self.colors = [pick_color(label, mode, bias_strength)
                       for _, label in base_dataset]

    def __len__(self):
        # Return how many images we have
        return len(self.base)

    def __getitem__(self, idx):
        # Grab the original grayscale image and its label
        img, label = self.base[idx]       
        # Squeeze out the channel dimension and convert to numpy
        gray = img.squeeze().numpy()      
        # Grab the color we assigned to this specific image
        c = self.colors[idx]
        
        # Colorize the image by multiplying the grayscale value with the RGB color
        # This keeps the background black since 0 * color = 0
        rgb = np.stack([gray * c[0], gray * c[1], gray * c[2]], axis=0)
        
        # Return it as a PyTorch tensor
        return torch.FloatTensor(rgb), label

class CNN(nn.Module):
    """Simple 2-block CNN."""

    def __init__(self, in_channels=3):
        super().__init__()
        # Define a very basic CNN with 2 convolutional blocks and some linear layers at the end
        self.net = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           nn.ReLU(), nn.MaxPool2d(2),
            # Flatten it for the dense layers
            nn.Flatten(),
            # Dense layers
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # Just pass the input through the network
        return self.net(x)

class TinyViT(nn.Module):
    """
    Small Vision Transformer for 28x28 images.
    patch_size=7 -> 4x4 = 16 patches per image.
    """

    def __init__(self, in_channels=3, patch_size=7, embed_dim=64, num_heads=4, num_layers=4):
        super().__init__()
        # Calculate how many patches we'll have based on the image size (28x28)
        num_patches = (28 // patch_size) ** 2  
        
        # Use a convolutional layer to slice the image into patches and embed them at the same time
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        
        # Create a learnable classification token
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Create learnable positional embeddings so the model knows where each patch came from
        self.pos_embed  = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        # Set up one layer of the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True, 
        )
        # Stack multiple layers to build the full transformer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # A simple linear layer to map the final CLS token to our 10 digit classes
        self.head = nn.Linear(embed_dim, 10)

    def forward(self, x):
        # Get the batch size
        B = x.shape[0]
        
        # Turn image into patch embeddings and flatten/transpose them for the transformer
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  
        
        # Expand the CLS token to match the batch size
        cls = self.cls_token.expand(B, -1, -1)
        
        # Concatenate the CLS token with the patches and add positional embeddings
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        
        # Pass everything through the transformer
        x = self.transformer(x)
        
        # Only use the output of the CLS token (index 0) for classification
        return self.head(x[:, 0])  

def train(model, loader, epochs):
    # Move the model to the GPU if we have one
    model.to(DEVICE)
    # Set up the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Use standard cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss()

    # Loop through the dataset for the specified number of epochs
    for epoch in range(epochs):
        # Put the model in training mode
        model.train()
        total_loss = 0.0
        
        # Loop through batches of images and labels
        for imgs, labels in loader:
            # Move data to the device
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # Clear out the old gradients
            optimizer.zero_grad()
            
            # Get the predictions and calculate the loss
            loss = criterion(model(imgs), labels)
            
            # Backpropagate the error
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Keep track of the loss so we can print the average later
            total_loss += loss.item()
            
        # Print the average loss for this epoch
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")

def get_accuracy(model, loader):
    # Put the model in evaluation mode (disables dropout, etc.)
    model.eval()
    correct, total = 0, 0
    
    # We don't need to track gradients for evaluation, which saves memory
    with torch.no_grad():
        for imgs, labels in loader:
            # Move data to the device
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # See if the predicted class matches the actual label
            correct += (model(imgs).argmax(dim=1) == labels).sum().item()
            total   += len(labels)
            
    # Return the percentage correct
    return correct / total


def shape_vs_color_diagnostic(model_name, model, correct_color_loader, reversed_color_loader):
    """
    Tests the model on images with correct color vs reversed color.

    If accuracy drops a lot when the color is reversed, the model learned
    to use color as a shortcut rather than actually reading the digit shape.
    """
    # Test how it does when the colors are what it expects
    acc_correct  = get_accuracy(model, correct_color_loader)
    # Test how it does when we trick it by reversing the colors
    acc_reversed = get_accuracy(model, reversed_color_loader)
    
    # Calculate how much the accuracy tanked
    drop = acc_correct - acc_reversed

    # Print the results nicely
    print(f"\n{model_name}:")
    print(f"  Accuracy -- correct color : {acc_correct:.1%}")
    print(f"  Accuracy -- reversed color: {acc_reversed:.1%}")
    print(f"  Accuracy drop             : {drop:.1%}")

    # Give a human-readable verdict based on how big the drop was
    if drop > 0.25:
        verdict = "Relies HEAVILY on color -> shortcut learner"
    elif drop > 0.08:
        verdict = "Uses both color AND shape"
    else:
        verdict = "Relies mostly on digit shape -> robust"
        
    print(f"  Verdict                   : {verdict}")

    return acc_correct, acc_reversed, drop


def main():
    # Set random seeds so our results are reproducible
    random.seed(42)
    torch.manual_seed(42)

    # Define a transform to just convert the images to tensors
    to_tensor = transforms.ToTensor()
    
    # Download and load the standard training set
    train_raw = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=to_tensor)
    
    # Download and load the standard test set
    test_raw  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_tensor)

    print("\nBuilding colored datasets (assigns colors once, then caches them)...")
    
    # Create the biased training dataset where 90% of digits have their assigned color
    train_biased  = ColoredMNIST(train_raw, mode="biased",   bias_strength=BIAS_STRENGTH)
    
    # Create the test sets for all 3 of our evaluation conditions
    test_biased   = ColoredMNIST(test_raw,  mode="biased",   bias_strength=BIAS_STRENGTH)
    test_random   = ColoredMNIST(test_raw,  mode="random",   bias_strength=BIAS_STRENGTH)
    test_reversed = ColoredMNIST(test_raw,  mode="reversed", bias_strength=BIAS_STRENGTH)

    # Wrap the datasets in DataLoaders so we can iterate over them in batches
    train_colored_loader  = DataLoader(train_biased,  batch_size=BATCH_SIZE, shuffle=True)
    test_biased_loader    = DataLoader(test_biased,   batch_size=BATCH_SIZE)
    test_random_loader    = DataLoader(test_random,   batch_size=BATCH_SIZE)
    test_reversed_loader  = DataLoader(test_reversed, batch_size=BATCH_SIZE)

    # Also make DataLoaders for the plain grayscale version for comparison
    train_gray_loader = DataLoader(train_raw, batch_size=BATCH_SIZE, shuffle=True)
    test_gray_loader  = DataLoader(test_raw,  batch_size=BATCH_SIZE)

    print("\n" + "=" * 60)
    print(f"Training CNN on COLOR-BIASED data (bias={BIAS_STRENGTH})...")
    
    # Initialize our CNN with 3 input channels (RGB)
    cnn_color = CNN(in_channels=3)
    # Train it!
    train(cnn_color, train_colored_loader, EPOCHS_CNN)

    print(f"\nTraining ViT on COLOR-BIASED data (bias={BIAS_STRENGTH})...")
    
    # Initialize our Vision Transformer with 3 input channels (RGB)
    vit_color = TinyViT(in_channels=3)
    # Train it!
    train(vit_color, train_colored_loader, EPOCHS_VIT)

    print("\nTraining CNN on GRAYSCALE data (no color)...")
    
    # Initialize a CNN but with only 1 input channel since it's grayscale
    cnn_gray = CNN(in_channels=1)
    # Train it!
    train(cnn_gray, train_gray_loader, EPOCHS_CNN)

    print("\nTraining ViT on GRAYSCALE data (no color)...")
    
    # Initialize a ViT but with only 1 input channel since it's grayscale
    vit_gray = TinyViT(in_channels=1)
    # Train it!
    train(vit_gray, train_gray_loader, EPOCHS_VIT)

    print("\n" + "=" * 60)
    print("Evaluating...")

    # Build a dictionary to store all our test results
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

    # Grayscale models don't see color at all, so all 3 conditions give the exact same result. 
    # We test once on plain grayscale and just repeat it across all columns for the table.
    gray_cnn_acc = get_accuracy(cnn_gray, test_gray_loader)
    gray_vit_acc = get_accuracy(vit_gray, test_gray_loader)

    # Insert the grayscale results into our dictionary
    results["CNN (grayscale)"] = {c: gray_cnn_acc for c in ["Same bias (0.9)", "Random color", "Reversed color"]}
    results["ViT (grayscale)"] = {c: gray_vit_acc for c in ["Same bias (0.9)", "Random color", "Reversed color"]}

    # Print out a nice summary table with all the numbers
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

    # Run the diagnostic to figure out if the color models are shortcut learners
    print("\n" + "=" * 65)
    print("SHAPE VS COLOR DIAGNOSTIC")
    print("If a model relies on color instead of digit shape, its accuracy")
    print("will drop sharply when color labels are reversed.")
    print("=" * 65)

    shape_vs_color_diagnostic("CNN (color)", cnn_color, test_biased_loader, test_reversed_loader)
    shape_vs_color_diagnostic("ViT (color)", vit_color, test_biased_loader, test_reversed_loader)

    print("\nDone!")

# This makes sure the main function only runs if we execute this script directly
if __name__ == "__main__":
    main()
