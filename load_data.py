# Load all dataset for training/testing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_Omniglot():
    # Define transformations: resize to 96x96 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load background (train) and evaluation (test) sets
    train_dataset = datasets.Omniglot (root="./data", background=True, download=True, transform=transform)
    test_dataset = datasets.Omniglot(root="./data", background=False, download=True, transform=transform)

    print(f"Training samples: {len(train_dataset)}, Classes: {len(train_dataset._characters)}")

    return train_dataset, test_dataset