import os

import torch
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image
import numpy as np
import json

from torch.utils.data import DataLoader, Dataset


class PlantNet:
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 16,
                 is_preprocessed=True,
                 image_size: int = None,
                 num_workers: int = 0):
        """
        Wrapper to load, preprocess and de-process PlantNet300k dataset.

        Args:
            data_dir: Directory of PlantNet 300K dataset (including the species names json)
            batch_size: Batch size
            image_size: Desired image size for data
            num_workers: Number of workers to load images
        """
        if not os.path.exists(data_dir):
            raise ValueError(f'Path "{data_dir}" does not exist')

        class_file_path = os.path.join(data_dir, 'plantnet300K_species_names.json')
        if not os.path.exists(class_file_path):
            raise ValueError(f'Species name file {class_file_path} does not exist')

        class_file = open(class_file_path, 'r')
        self.class_to_name = json.load(class_file)
        class_file.close()

        self.classes = list(self.class_to_name.keys())
        self.n_classes = len(self.classes)

        data_dir_train = os.path.join(data_dir, "images_train")
        data_dir_val = os.path.join(data_dir, "images_val")
        data_dir_test = os.path.join(data_dir, "images_test")

        self.batch_size = batch_size
        self.is_preprocessed = is_preprocessed
        self.image_size = image_size

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

        self.train_set = PlantNetDataset(data_root=data_dir_train, classes_list=self.classes,
                                         is_preprocessed=self.is_preprocessed, image_size=self.image_size)
        self.train_set.transform = self.train_transform
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)

        self.val_set = PlantNetDataset(data_root=data_dir_val, classes_list=self.classes,
                                       is_preprocessed=self.is_preprocessed, image_size=self.image_size)
        self.val_set.transform = self.val_transform
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size,
                                     shuffle=False, num_workers=num_workers)

        self.test_set = PlantNetDataset(data_root=data_dir_test, classes_list=self.classes,
                                        is_preprocessed=self.is_preprocessed, image_size=self.image_size)
        self.test_set.transform = self.test_transform
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size,
                                      shuffle=True, num_workers=num_workers)

        # invert normalization for tensor to image transform
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean=0, std=[1. / s for s in self.std]),
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
    def __init__(self,
                 data_root,
                 classes_list,
                 is_preprocessed=False,
                 image_size=None,
                 transform=None):
        self.data_root = data_root

        self.data = []

        self.is_preprocessed = is_preprocessed
        self.image_size = image_size

        self.transform = transform if transform is not None else transforms.ToTensor()

        for class_folder in os.listdir(self.data_root):
            class_path = os.path.join(self.data_root, class_folder)

            # we are only interested in directories
            if not os.path.isdir(class_path):
                continue

            label_idx = classes_list.index(class_folder)

            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)

                # one data sample is full filepath and index in classes_list
                self.data.append((file_path, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        img = self.pil_loader(img_path)
        img = self.transform(img)

        return img, label

    def pil_loader(self, path: str):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)

            # if size is given, we make the image a square image in the given size
            if self.image_size and not self.is_preprocessed:
                # make non-square images square
                if img.width != img.height:
                    new_size = min(img.width, img.height)

                    left = int((img.width - new_size) / 2)
                    top = int((img.height - new_size) / 2)
                    right = int((img.width + new_size) / 2)
                    bottom = int((img.height + new_size) / 2)

                    img = img.crop((left, top, right, bottom))

                img = img.resize((self.image_size, self.image_size))

            elif not self.is_preprocessed:
                raise ValueError("is_preprocessed flag is set to false but no image size for preprocessing is given!")

            return np.array(img)


if __name__ == "__main__":
    # plantnet_dir = 'D:\data'
    plantnet_dir = '/media/johannes-f/F038E25338E217FC/Users/Merkur-PC/Desktop/Plantnet300K/plantnet_300K/'

    data = PlantNet(data_dir=plantnet_dir,
                    batch_size=16,
                    image_size=128)

    for ims, labels in data.train:
        print("images")
        print("\t", ims.shape)
        print(f"\t {ims.min()} < {torch.mean(ims)} < {ims.max()}")
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
