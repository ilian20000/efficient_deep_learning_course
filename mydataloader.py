from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch

# Transformation functions

normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=(0, 15)),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

# Loading datasets

debug_size = 4
rootdir = './data/cifar10'

def load_trainset(half=False, debug=False, augment=2):
    if augment==0:
        transform_type = transform_test
    elif augment==1:
        transform_type =  transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])
    else:
        transform_type=transform_train
    if half:
        transform_type = transforms.Compose([
            transform_type,
            transforms.Lambda(lambda x: x.half())
        ]) 

    n_workers = 4
    c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_type)
    if debug :
        num_samples_subset = 32*debug_size
        indices_train = list(range(len(c10train)))
        seed  = 2147483647
        np.random.RandomState(seed=seed).shuffle(indices_train) ## modifies the list in place
        c10train = torch.utils.data.Subset(c10train,indices_train[:num_samples_subset])

    trainloader = DataLoader(c10train, batch_size=32,shuffle=True, num_workers=n_workers) # A FINETUNE
    return trainloader


def load_testset(half=False, debug=False):
    if half:
        transform_type = transforms.Compose([
            transform_test,
            transforms.Lambda(lambda x: x.half())
        ])
    else:
        transform_type=transform_test

    
    c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_type)
    if debug:
        num_samples_subset = 32*debug_size
        indices_test = list(range(len(c10test)))
        seed  = 2147483647
        np.random.RandomState(seed=seed).shuffle(indices_test) ## modifies the list in place
        c10test = torch.utils.data.Subset(c10test,indices_test[:num_samples_subset])
    
    testloader = DataLoader(c10test, batch_size=32, shuffle=True) 
    return testloader



