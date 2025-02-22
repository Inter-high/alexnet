"""
This script provides a custom dataset and DataLoader for the CIFAR-10 dataset with data augmentation. 
It includes train and test transformations, dataset splitting, and DataLoader creation.
The transformations apply cropping, resizing, flipping, and normalization.
A custom collate function ensures correct handling of multiple augmented images per sample.

Author: yumemonzo@gmail.com
Date: 2025-02-21
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List


class Cifar10(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True):
        """
        Custom wrapper for CIFAR-10 dataset.

        Args:
            root (str): Directory where dataset is stored.
            train (bool): Whether to load training set.
            transform: Transformations to apply.
            download (bool): Whether to download dataset if not found.
        """
        self.dataset = CIFAR10(
            root=root, train=train, transform=transform, download=download
        )

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Retrieves an image and its label by index."""
        image, label = self.dataset[index]
        return image, label
    

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Defines image transformations for training and testing.

    Returns:
        Tuple containing training and testing transformations.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.Lambda(lambda img: [img, img.transpose(Image.FLIP_LEFT_RIGHT)]),  # Original + flipped
        transforms.Lambda(lambda imgs: [transforms.ToTensor()(img) for img in imgs]),
        transforms.Lambda(lambda imgs: [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])(img) for img in imgs])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),  # Generates 10 cropped images
        transforms.Lambda(lambda crops: crops + tuple(crop.transpose(Image.FLIP_LEFT_RIGHT) for crop in crops)),  # Add flipped crops
        transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
        transforms.Lambda(lambda crops: [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])(crop) for crop in crops])
    ])

    return train_transform, test_transform


def get_datasets(data_dir: str) -> Tuple[Cifar10, Cifar10]:
    """
    Loads CIFAR-10 training and testing datasets with transformations.

    Args:
        data_dir (str): Directory path for CIFAR-10 dataset.

    Returns:
        Tuple containing training and testing datasets.
    """
    train_transform, test_transform = get_transforms()

    dataset = Cifar10(root=data_dir, train=True, transform=train_transform, download=True)
    test_dataset = Cifar10(root=data_dir, train=False, transform=test_transform, download=True)

    return dataset, test_dataset


def split_dataset(seed: int, dataset: Dataset, ratio: List[float] = [0.8, 0.2]) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and validation sets.

    Args:
        seed (int): Random seed for reproducibility.
        dataset (Dataset): Dataset to be split.
        ratio (List[float]): Proportions for training and validation sets.

    Returns:
        Tuple containing training and validation datasets.
    """
    generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset = random_split(dataset, ratio, generator)

    return train_dataset, valid_dataset


def collate_fn(batch: List[Tuple[List[torch.Tensor], int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle multiple transformed images per sample.

    Args:
        batch (List[Tuple[List[torch.Tensor], int]]): List of image-label pairs.

    Returns:
        Tuple containing stacked images and repeated labels.
    """
    images, labels = zip(*batch)
    
    images = [img for img_list in images for img in img_list]  # Flatten image list
    num_transforms = len(images) // len(labels)  # Determine number of transformations per image
    labels = [label for label in labels for _ in range(num_transforms)]  # Repeat labels accordingly

    return torch.stack(images), torch.tensor(labels)


def get_train_loader(train_dataset: Dataset, valid_dataset: Dataset, 
                     batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for training and validation.

    Args:
        train_dataset (Dataset): Training dataset.
        valid_dataset (Dataset): Validation dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads.

    Returns:
        Tuple containing training and validation DataLoaders.
    """
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader


def get_test_loader(test_dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    """
    Creates a DataLoader for testing.

    Args:
        test_dataset (Dataset): Testing dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader for the test dataset.
    """
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return test_dataloader
