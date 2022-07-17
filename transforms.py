import torch
import torchvision.transforms as transforms

# basic transformations + augmentation
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)),
    transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
    ])

# basic transformations
test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)),
        transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
        ])