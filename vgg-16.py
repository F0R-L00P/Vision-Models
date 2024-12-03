# bsic libraries
import numpy as np
import matplotlib as plt

# DL libraries
import torch
import torch.nn
import torchvision
from torch.utils.data import DataLoader

# designed libraries
from helper_functions import *
#######################
###### Settings#########
#######################

random_seed = 1818
batch_size = 32
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

# deterministic seed
set_seed(random_seed)

#######################
# using CIFAR-10 dataset
#######################
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((64, 64)),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    torchvision.transforms.ToTensor()
])


test_transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((64, 64)),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(
    root=r'C:\Users\behna\Documents\Vision-Models\data',
    train=True,
    download=True,
    transform=train_transform
)

test_set = torchvision.datasets.CIFAR10(
    root=r'C:\Users\behna\Documents\Vision-Models\data',
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8
)

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
        super()