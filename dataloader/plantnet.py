import os

import torch
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image
import numpy as np
import json

from torch.utils.data import DataLoader, Dataset


class PlantNet:
    def __init__(self, class_file_path: str, data_dir: str, batch_size: int = 16):
        """
        Wrapper to load, preprocess and deprocess PlantNet300k dataset.
        Args:
            batch_size (int): Batch size, default: 16.
        """
        if not os.path.exists(data_dir):
            raise ValueError(f'Path "{data_dir}" does not exist')

        self.batch_size = batch_size

        class_file = open(class_file_path, 'r')
        self.class_to_name = json.load(class_file)
        class_file.close()

        self.classes = list(self.class_to_name.keys())
        self.n_classes = len(self.classes)

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.mean, self.std)
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        dataset_full = PlantNetDataset(data_root=data_dir, classes_list=self.classes)

        train_size = int(0.7 * len(dataset_full))
        val_size = int(0.2 * len(dataset_full))
        test_size = len(dataset_full) - train_size - val_size

        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(dataset_full,
                                                                                    [train_size, val_size, test_size])

        self.train_set.dataset.transform = self.train_transform
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        self.val_set.dataset.transform = self.val_transform
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

        self.test_set.dataset.transform = self.test_transform
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

        # invert normalization for tensor to image transform
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean=0, std=[1./s for s in self.std]),
            transforms.Normalize(mean=[-m for m in self.mean], std=1.),
            lambda x: x * 255
        ])

    @property
    def train(self):
        """ Return training dataloader. """
        return self.train_loader

    @property
    def val(self):
        """ Return validation dataloader. """
        return self.val_loader

    @property
    def test(self):
        """ Return test dataloader. """
        return self.test_loader

    def idx2label(self, idx):
        """ Return class label for given index. """
        class_name = self.classes[idx]
        return self.class_to_name[class_name]

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
    def __init__(self, data_root, classes_list, transform=None):
        self.data_root = data_root

        self.data = []

        self.transform = transform if transform is not None else transforms.ToTensor()

        for class_folder in os.listdir(self.data_root):
            class_path = os.path.join(self.data_root, class_folder)

            label_idx = classes_list.index(class_folder)

            for file in os.listdir(class_path):

                file_path = os.path.join(class_path, file)

                # one datasample is full filepath and index in classes_list
                self.data.append((file_path, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        img = self.pil_loader(img_path)
        img = self.transform(img)

        return img, label

    @staticmethod
    def pil_loader(path: str):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return np.array(img)


if __name__ == "__main__":
    plantnet_dir = '/home/johannes-f/OneDrive/Computer Vision & Visual Synthesis/project/code/big_data/PlantNet_64'

    data = PlantNet(class_file_path='plantnet_300K_species_names.json',
                    data_dir=plantnet_dir,
                    batch_size=16)

    for ims, labels in data.train:
        print("images")
        print("\t", ims.shape)
        print(f"\t {ims.min()} < {torch.mean(ims)} < {ims.min()}")
        print("labels")
        print("\t", labels)
        print("\t", [data.idx2label(i) for i in labels])

        ims = ims.detach().cpu()
        ims_grid = torchvision.utils.make_grid(ims)

        pil_ims = data.tensor2img(ims_grid)

        SAVE_DIR = 'datasamples'
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        pil_ims.save(os.path.join(SAVE_DIR, 'plantnet_sample.png'))

        break
