import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import json

from torch.utils.data import DataLoader, Dataset


class PlantNet:
    def __init__(self, batch_size: int = 16):
        """
        Wrapper to load, preprocess and deprocess PlantNet300k dataset.
        Args:
            batch_size (int): Batch size, default: 16.
        """

        self.batch_size = batch_size

        class_file = open("data/plantnet300K_species_names.json")
        class_to_name = json.load(class_file)
        class_file.close()
        self.classes = list(class_to_name.keys())
        self.n_classes = len(self.classes)

        # TODO: change these values?
        self.mean = [0.491, 0.482, 0.446]
        self.std = [1., 1., 1.]

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # TODO: this has to be read from a config file
        dataset_full = PlantNetDataset(data_root="D:\\data\\test_data_dst\\")

        train_size = int(0.8 * len(dataset_full))
        val_size = len(dataset_full) - train_size

        self.train_set, self.val_set = torch.utils.data.random_split(dataset_full, [train_size, val_size])

        self.train_set.dataset.transform = self.train_transform
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        self.val_set.dataset.transform = self.val_transform
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

        # invert normalization for tensor to image transform
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean=0, std=[1./s for s in self.std]),
            transforms.Normalize(mean=[-m for m in self.mean], std=1.)
        ])

    @property
    def train(self):
        """ Return training dataloader. """
        return self.train_loader

    @property
    def val(self):
        """ Return validation dataloader. """
        return self.val_loader

    def idx2label(self, idx):
        """ Return class label for given index. """
        return self.classes[idx]

    def prob2label(self, prob_vector):
        """ Return class label with highest confidence. """
        return self.idx2label(torch.argmax(prob_vector).cpu())

    def tensor2img(self, tensor):
        """ Convert torch.Tensor to PIL image. """
        n_channels = tensor.shape[0]

        img = tensor.detach().cpu()
        img = self.inv_normalize(img)

        if n_channels > 1:
            return Image.fromarray(img.permute(1, 2, 0).numpy().astype('uint8')).convert("RGB")
        else:
            return Image.fromarray(img[0].numpy()).convert("L")


class PlantNetDataset(Dataset):
    def __init__(self, data_root, transform=None):

        self.data_root = data_root
        self.transform = transform

        self.data = []

        for subdir, dirs, files in os.walk(self.data_root):

            label = subdir.replace(self.data_root, "")

            for file in files:
                # filename, class_label
                self.data.append((file, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.data[idx][1]

        img = Image.open(os.path.join(self.data_root, label, self.data[idx][0]))

        if self.transform:
            img = self.transform(img)

        return img, label
