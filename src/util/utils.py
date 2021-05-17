"""
University of Sherbrooke
PhD Project
Authors: D'Jeff Kanda
"""
import os

import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F


def get_data(data_augment: bool, dataset: str = 'mnistfashion'):
    """
    This function loads the dataset if it already exists, otherwise it downloads from pytorch dataset repository.

    :param data_augment: if true, data transformation will be applied on the dataset
    :param dataset: the name of the dataset to load or download ['mnist' or 'cifar100']
    :return: train set and test set
    """
    if data_augment:
        print('Data augmentation activated!')
        if dataset == 'cifar100':
            train_transform = transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ColorJitter(brightness=.2, contrast=.2, hue=.05,
                                       saturation=.05),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomRotation(2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
            base_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
    else:
        if dataset == 'cifar100':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
            base_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
    # Download the train and test set and apply transform on it
    if dataset == 'cifar100':
        train_set = datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR100(root='../data', train=False, download=True, transform=base_transform)
    else:
        train_set = datasets.FashionMNIST(root='../data', train=True, download=True, transform=train_transform)
        test_set = datasets.FashionMNIST(root='../data', train=False, download=True, transform=base_transform)

    return train_set, test_set


def predict_proba(scores):
    """
    Predicts probability from the score
    :arg
        scores: the score values from the model
    """
    prob = F.softmax(scores, dim=1)
    return prob


def check_dir(path):
    """
    This function ensure that a path exists or create it
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
