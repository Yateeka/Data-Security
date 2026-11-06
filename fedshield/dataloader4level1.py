import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn


def dataloader(train_dataset, test_dataset):

    # TODO: Set the length of the batch (number of samples per batch)
    batch_size = 50

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
   
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=True
    )
    print(f'training has：{len(train_loader)} batch of data！')
    # print(f'validation has：{len(vali_loader)} batch of data！')
    print(f'testing has：{len(test_loader)} batch of data！')
    return train_loader, test_loader


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)

    print("The number of training data：", len(train_dataset))
    print("The number of testing data：", len(test_dataset))

    return dataloader(train_dataset, test_dataset)