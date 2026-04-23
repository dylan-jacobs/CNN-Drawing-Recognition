"""
Authors: Dylan Jacobs, Jonah Pflaster, Pablo Velarde
Date: 30 April, 2026
"""

# %% 1- Load and Preprocess the Data
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,], std=[0.5,]) # I think images in black and white
)

