import torch

from data.cifar10 import CIFAR10
from data.plantnet import PlantNet

if __name__ == "__main__":

    data = PlantNet()

    train_loader = data.train
    val_loader = data.val
    test_loader = data.test

    print(len(train_loader) * 16)
    print(len(val_loader) * 16)
    print(len(test_loader) * 16)