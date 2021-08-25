import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config import arg



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def train_loader():
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=arg.batch_size, shuffle=True,
            num_workers=arg.num_workers, pin_memory=True)
    return train_loader

def val_loader():
    val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=arg.batch_size, shuffle=False,
            num_workers=arg.num_workers, pin_memory=True)
    return val_loader