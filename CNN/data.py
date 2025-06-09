import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def get_data(batch_size, valid_ratio, seed):
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    valid_size = int(len(full_trainset) * valid_ratio)
    train_size = len(full_trainset) - valid_size
    train_subset, val_subset = random_split(
        full_trainset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed)
    )
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, validloader, testloader
