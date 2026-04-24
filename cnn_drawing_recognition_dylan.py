"""
Authors: Dylan Jacobs, Jonah Pflaster, Pablo Velarde
Date: 30 April, 2026
"""


# %% 1- Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import random_split
from google.colab import drive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mount drive to access dataset from Colab
drive.mount('/content/drive')

# %% - Load and Preprocess data

batch_size = 64
root = "/content/drive/MyDrive/Colab Notebooks/png_ready"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,], std=[0.5,]) # I think images in black and white
])

dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)

# split into train, validation, test
train_test_split = 0.8
train_val_split = 0.8
full_dataset_len = len(dataset)
full_train_len = int(train_test_split * len(dataset))
test_len = full_dataset_len - full_train_len
[full_trainset, testset] = random_split(dataset, [full_train_len, test_len])

# split train into train/val
train_len = int(train_val_split * full_train_len)
val_len = full_train_len - train_len
[trainset, valset] = random_split(full_trainset, [train_len, val_len])

print(f'Training length: ', train_len)
print(f'Validation length: ', val_len)
print(f'Testing length: ', test_len)

# Create torch dataloaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# %% Preview images
def preview_images(data_loader, classes):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    for i, ax in enumerate(axes):
        img = images[i]
        img = img.permute(1, 2, 0).numpy() # reshape into correct size for image
        ax.imshow(img)
        ax.set_title(classes[labels[i].item()])
        ax.axis('off')
    plt.show()

classes = dataset.classes
preview_images(train_loader, classes)

# %% 
