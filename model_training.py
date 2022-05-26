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

        if label == 1:
            label = torch.Tensor([0, 1])
        elif label == 0:
            label = torch.Tensor([1, 0])

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

        x = torch.squeeze(x, dim=0)        
        features = self.pretrained_model(x)        
        features = torch.unsqueeze(features, dim=0)
        
        features_avg = self.avg_pooling_layer(features)
        features_max = self.max_pooling_layer(features)

        features_avg = self.flatten(features_avg)
        features_max = self.flatten(features_max)
        
        features_avg = torch.squeeze(features_avg, dim=0)
        features_max = torch.squeeze(features_max, dim=0)
        features_concat = torch.cat((features_avg, features_max), dim=0)
        features_concat = torch.unsqueeze(features_concat, dim=0)
        
        output = self.classifier(features_concat)
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
    len_train_dataset = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # model, optimizer, criterion
    model = MriNet(pretrained_model_type)
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # set model to training mode
    model.train()

    for epoch in range(n_epochs):

        for state in ["train", "test"]:

            logging.info(f"Epoch {epoch}, State: {state}")

            running_loss = 0.0
            running_corrects = 0
            
            if state == "train":
                
                # dataset and loader
                dataset = MriDataset(root_dir, True, view_type, abnormality_type, transform = data_transforms)
                len_dataset = len(dataset)
                dataloader = DataLoader(dataset, batch_size, shuffle=True)
                model.train()
            
            else:

                # dataset and loader
                dataset = MriDataset(root_dir, False, view_type, abnormality_type, transform = data_transforms)
                len_dataset = len(dataset)
                dataloader = DataLoader(dataset, batch_size, shuffle=True)
                model.eval()

            for id, batch in enumerate(dataloader, 0):
                logging.info(f"Batch: {batch}")
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # calculate loss
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if state == "train":
                    loss.backward()
                    optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            # print epoch statistics
            epoch_loss = round(running_loss / len_dataset, 2)
            epoch_acc = round(running_corrects / len_dataset, 2)
            print("Loss: {epoch_loss}, accuracy: {epoch_acc}")

    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = train_model(device, args["root_dir"], args["view_type"], args["abnormality_type"], 
                            args["pretrained_model_type"], args["batch_size"], args["n_epochs"])