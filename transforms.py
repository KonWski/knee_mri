import torch
import torchvision.transforms as transforms
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine

# transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
# transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))

# basic transformations + augmentation
train_transforms = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.RandomRotation(20),
    # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    # transforms.RandomHorizontalFlip(),
    # transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),    
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)),
    # transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
    
    transforms.Lambda(lambda x: torch.Tensor(x)),
    RandomRotate(25),
    RandomTranslate([0.11, 0.11]),
    RandomFlip(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),

    ])



# basic transformations
test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
        # transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
        # transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)),
        transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
        ])