import torch
import torchvision.transforms as transforms

# basic transformations + augmentation
train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: torch.Tensor(x)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3))
    ])

# basic transformations
test_transforms = transforms.Compose([
    transforms.Lambda(lambda x: torch.Tensor(x)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3))
    ])