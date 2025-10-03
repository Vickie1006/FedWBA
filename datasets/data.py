# FedWBA: Personalized Bayesian Federated Learning with Wasserstein Barycenter Aggregation
# Copyright (C) 2025  Ting Wei

import torch
import torchvision
import torchvision.transforms as transforms
from collections import Counter
import numpy as np
from torch.utils.data import DataLoader


def data_split(num_devices, num_labels, dataset_type="MNIST", check=False):
   
    """
    Read a specified dataset (MNIST, FashionMNIST, CIFAR10, or CIFAR100) from torchvision, perform data distribution, and return the data in Tensor type.

    Parameters:
    num_devices: The number of clients
    server_samples: The number of samples to be drawn from the training set for the server (actually used for testing only, not for training)
    dataset_type: The type of dataset to be loaded. Optional values are "MNIST", "FashionMNIST", "CIFAR10", or "CIFAR100"
    check: Whether to check the distributed data and print relevant information. The default is False

    Returns:
    client_data: A dictionary containing client data. The keys are client indices, and the values are corresponding data tuples (training images, training labels, test images, test labels), all of which are of type torch.Tensor
    """

    
    if dataset_type == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        max_label = 10
    elif dataset_type == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        max_label = 10
    elif dataset_type == "CIFAR10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        max_label = 10
    elif dataset_type == "CIFAR100":
       
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        max_label = 20 
       
        superclass = [
            [4, 30, 55, 72, 95],
            [1, 32, 67, 73, 91],
            [54, 62, 70, 82, 92],
            [9, 10, 16, 28, 61],
            [0, 51, 53, 57, 83],
            [22, 39, 40, 86, 87],
            [5, 20, 25, 84, 94],
            [6, 7, 14, 18, 24],
            [3, 42, 43, 88, 97],
            [12, 17, 37, 68, 76],
            [23, 33, 49, 60, 71],
            [15, 19, 21, 31, 38],
            [34, 63, 64, 66, 75],
            [26, 45, 77, 79, 99],
            [2, 11, 35, 46, 98],
            [27, 29, 44, 78, 93],
            [36, 50, 65, 74, 80],
            [47, 52, 56, 59, 96],
            [8, 13, 48, 58, 90],
            [41, 69, 81, 85, 89]
        ]
    else:
        raise ValueError("Invalid dataset_type. Choose either 'MNIST', 'FashionMNIST', 'CIFAR10' or 'CIFAR100'.")

    client_data = {}
    server_data = {}

    train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)
    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=True)

    train_imgs, train_labels = next(iter(train_loader))
    test_imgs, test_labels = next(iter(test_loader))

    train_permutation = torch.randperm(train_labels.shape[0])
    train_imgs = train_imgs[train_permutation]
    train_labels = train_labels[train_permutation]

    test_permutation = torch.randperm(test_labels.shape[0])
    test_imgs = test_imgs[test_permutation]
    test_labels = test_labels[test_permutation]

    
    for client_idx in range(num_devices):
        if dataset_type == "CIFAR100":
            superclass_indices = [client_idx % max_label, (client_idx + 1) % max_label]
            client_labels = []
            for idx in superclass_indices:
                client_labels.extend(superclass[idx])
        elif dataset_type in ["MNIST", "FashionMNIST", "CIFAR10"]:
            client_labels = [((client_idx * num_labels + j) % max_label) for j in range(num_labels)]

        train_mask = torch.isin(train_labels, torch.tensor(client_labels))
        test_mask = torch.isin(test_labels, torch.tensor(client_labels))

        train_indices = torch.where(train_mask)[0][:10000]
        test_indices = torch.where(test_mask)[0][:10000 // 4]

        client_train_imgs = train_imgs[train_indices]
        client_train_labels = train_labels[train_indices]
        client_test_imgs = test_imgs[test_indices]
        client_test_labels = test_labels[test_indices]

        client_data[client_idx] = (
            client_train_imgs,
            client_train_labels,
            client_test_imgs,
            client_test_labels
        )

   
    if check:
        print("Client data info:")
        for client_idx, (X_train, y_train, X_test, y_test) in client_data.items():
            print(f"Client {client_idx}")
            label_count_client = Counter(y_train.tolist())
            unique_labels_client = y_train.unique()
            unique_labels_str = ', '.join([str(label.item()) for label in unique_labels_client])
            print(unique_labels_str)
            print('train')
            print(', '.join([f'{label}: {count}' for label, count in label_count_client.items()]))
            print('test')
            label_count_client = Counter(y_test.tolist())
            unique_labels_client = y_test.unique()
            unique_labels_str = ', '.join([str(label.item()) for label in unique_labels_client])
            print(', '.join([f'{label}: {count}' for label, count in label_count_client.items()]))
            
        print('---------------------------------------')
        
    return client_data
