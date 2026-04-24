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
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mount drive to access dataset from Colab
drive.mount('/content/drive')

# %% - Load and Preprocess data

batch_size = 16
img_size = 64

root = "/content/drive/MyDrive/Colab Notebooks/png_ready"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=2, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, num_workers=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=2, shuffle=False)

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

# %% Define CNN class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, img_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(img_size, img_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(img_size, img_size, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 250)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = torch.flatten(x, 1) # need to flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc_out(x) # no need for relu bc torch does softmax automatically
        return x
    


# %% Train model

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)

def train_model(model, train_loader, val_loader, epochs=10):
    train_losses, val_losses = [], [] # track for loss curves

    for epoch in range(epochs):
        total_train_loss = 0.0
        total_val_loss   = 0.0

        # training step
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation step
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)

            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # update LR if necessary according to validation
        lr_scheduler.step(avg_val_loss)

        print(f'Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4}, Val Loss: {avg_val_loss:.4}, Learning Rate: {optimizer.param_groups[0]['lr']:.6}')
    return train_losses, val_losses


train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10)
print('Finished training')

# %%
