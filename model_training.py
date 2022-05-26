import os
import pandas as pd
import numpy as np
import logging
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
from pretrained_models import get_pretrained_model
import torchvision.transforms as transforms

def get_args():
    parser = argparse.ArgumentParser(description='Process paramaters for model learning')
    parser.add_argument('--view_type', type=str, help='axial/coronal/sagittal')
    parser.add_argument('--abnormality_type', type=str, help='abnormal/acl/meniscus')
    parser.add_argument('--root_dir', type=str, help='root_dir/view_type')
    parser.add_argument('--pretrained_model_type', type=str, help='Type of model used for feature extraction AlexNet/Resnet/Inception')
    parser.add_argument('--batch_size', type=int, help='Number of images in batch')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')

    args = vars(parser.parse_args())
    
    logging.info(8*"-")
    logging.info("PARAMETERS")
    logging.info(8*"-")
    for parameter in args.keys():
        logging.info(f"{parameter}: {args[parameter]}")
    logging.info(8*"-")

    return args


class MriDataset(data.Dataset):
    '''
    Attributes
    ----------
    root_dir : str
        path to root directory. Leads to train and val
    train: bool
        True -> root_dir/Train
        False -> root_dir/Val
    view_type: str
        axial/coronal/sagittal
    abnormality_type: str
        abnormal/acl/meniscus
    transform: 
        set of transformations used for image preprocessing
    '''

    def __init__(self, root_dir, train, view_type, abnormality_type, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.train = train
        self.view_type = view_type
        self.abnormality_type = abnormality_type

        subfolder = "train" if train else "valid"
        self.dataset_path = f"{self.root_dir}/{subfolder}/{view_type}"
        self.labels = pd.read_csv(f"{self.root_dir}/{subfolder}-abnormal.csv", 
                                      names=["id", "abnormality"], 
                                      dtype={"id": str, "abnormality": int})
        self.labels = self.labels.set_index("id")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        df_index = "0" * (4 - len(str(index))) + str(index)
        image = np.load(f"{self.dataset_path}/{df_index}.npy")
        label = self.labels.loc[df_index]["abnormality"]
        
        if self.transform:
            image = self.transform(image)

        return image, label


class MriNet(nn.Module):
    
    def __init__(self, pretrained_model_type):
        super().__init__()
        self.pretrained_model = get_pretrained_model(pretrained_model_type)
        self.avg_pooling_layer = nn.AdaptiveAvgPool2d((15, 15))
        self.max_pooling_layer = nn.AdaptiveMaxPool2d((15, 15))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(450, 2)

    def forward(self, x):

        print(f"X before squeeze: {x.size()}")
        x = torch.squeeze(x, dim=0)
        print(f"X after squeeze: {x.size()}")
        
        features = self.pretrained_model(x)
        print(f"features after pretrained_model: {features.size()}")
        
        features = torch.unsqueeze(features, dim=0)
        print(f"features after unsqueeze: {features.size()}")
        
        features_avg = self.avg_pooling_layer(features)
        features_max = self.max_pooling_layer(features)
        print(f"features_avg shape: {features_avg.shape}")
        print(f"features_max shape: {features_max.shape}")

        features_avg = self.flatten(features_avg)
        features_max = self.flatten(features_max)
        print(f"features_avg shape: {features_avg.shape}")
        print(f"features_max shape: {features_max.shape}")
        
        features_avg = torch.squeeze(features_avg, dim=0)
        features_max = torch.squeeze(features_max, dim=0)
        features_concat = torch.cat((features_avg, features_max), dim=0)
        print(f"features_concat shape: {features_concat.shape}")
        
        output = self.classifier(features_concat)

        pooled_features = self.pooling_layer(features)
        print(f"pooled_features after pooling_layer: {pooled_features.size()}")
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        print(f"pooled_features after view: {pooled_features.size()}")
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        print(f"flattened_features after view: {flattened_features.size()}")
        output = self.classifier(flattened_features)
        return output


def train_model(device, root_dir, view_type, abnormality_type, pretrained_model_type, batch_size, n_epochs):
    '''
    trains model for recognising selected abnormality on images taken from choosen view
    '''

    # transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
        transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
        ])

    # dataset and loader
    train_dataset = MriDataset(root_dir, True, view_type, abnormality_type, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # model, optimizer, criterion
    model = MriNet(pretrained_model_type)
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Set model to training mode
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(n_epochs):

        logging.info(f"Epoch {epoch}")
        running_loss = 0.0
        
        for id, batch in enumerate(train_loader, 0):
            
            images, labels = batch
            optimizer.zero_grad()

            # calculate loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            logging.info(f"Loss: {running_loss}")

    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = train_model(device, args["root_dir"], args["view_type"], args["abnormality_type"], 
                            args["pretrained_model_type"], args["batch_size"], args["n_epochs"])