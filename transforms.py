import torch
import torchvision.transforms as transforms

# basic transformations + augmentation
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
    transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
    transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
    ])

# basic transformations
test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
        transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
        ])