import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

import os
import sys
import time 
import glob
import argparse
import datetime
# Hyperparameters
num_epochs = 40
num_classes = 10
batch_size = 8
learning_rate = 1e-3
DATA_PATH = 'C:\\atrophyData\\data\\trainData\\'
MODEL_STORE_PATH = 'C:\\Users\\Andy\\PycharmProjects\\pytorch_models\\'

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor()])

# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
# test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
train_img = sorted(glob.glob(os.path.join(DATA_PATH, "*3DT1_Warped.nii")))
train_dataset = datasets.ImageFolder(root=DATA_PATH, transform=trans)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = ConvNet()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))