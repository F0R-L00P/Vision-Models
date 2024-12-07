# bsic libraries
import numpy as np
import matplotlib as plt
from tqdm import tqdm

from PIL import Image

# DL libraries
import torch
import torch.nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

# designed libraries
from helper_functions import train, validate, set_seed

#######################
###### Settings#########
#######################
random_seed = 1818
batch_size = 32
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

# Deterministic seed
set_seed(random_seed)

#######################
# Using CIFAR-10 dataset
#######################
train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop((32, 32)),  # Keep size consistent
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_transform = torchvision.transforms.Compose(
    [
        #        torchvision.transforms.CenterCrop((32, 32)),  # Keep size consistent
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

test_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Inspect DataLoader
print(f"Training set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")


########################
# VGG-16 Architecture
########################
class VGG16(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = torch.nn.Sequential(
            # BLOCK 1
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # BLOCK 2
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # BLOCK 3
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # BLOCK 4
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # BLOCK 5
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),  # Adjusted input size
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(f"Shape after features: {x.shape}")  # Debugging line
        x = torch.flatten(x, 1)  # Flatten the feature map
        # print(f"Shape after flatten: {x.shape}")  # Debugging line
        x = self.classifier(x)
        return x


###################### TEST
# Initialize the model
model = VGG16(num_classes=10).to(device)

# Generate random input
random_input = torch.randn(1, 3, 32, 32).to(device)

# Forward pass
output = model(random_input)

# Check the output
print("Output shape:", output.shape)  # Should be [1, 10]

#########################################
########### Training the model ##########
#########################################
# Define the model, loss, and optimizer
model = VGG16(num_classes=10).to(device)
criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss includes softmax
optimizer = optim.Adam(
    model.parameters(), lr=0.0001
)  # Adam optimizer with a learning rate

# Training loop
best_accuracy = 0.0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_loss, train_accuracy = train(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_accuracy = validate(model, test_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), "vgg16_best.pth")
        print("Best model saved!")

print(f"Training complete. Best validation accuracy: {best_accuracy:.2f}%")

# Create a checkpoint dictionary
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_accuracy": best_accuracy,
    # 'scheduler_state_dict': scheduler.state_dict()
}

# Save the checkpoint
torch.save(checkpoint, "vgg16_checkpoint.pth")
print("Training complete. Model and optimizer saved successfully.")
