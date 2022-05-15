import os
import pandas as pd
import numpy as np
import logging
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
from pretrained_models import get_pretrained_model

def get_args():
    parser = argparse.ArgumentParser(description='Process paramaters for model learning')
    parser.add_argument('--view_type', type=str, help='axial/coronal/sagittal')
    parser.add_argument('--abnormality_type', type=str, help='abnormal/acl/meniscus')
    parser.add_argument('--dataset_path', type=str, help='')
    parser.add_argument('--pretrained_model_type', type=str, help='Type of model used for feature extraction AlexNet/Resnet/Inception')
    parser.add_argument('--batchsize', type=int, help='')
    args = vars(parser.parse_args())
    
    logging.info(8*"-")
    logging.info("PARAMETERS")
    logging.info(8*"-")
    for parameter in args.keys():
        logging.info(f"{parameter}: {args[parameter]}")
    logging.info(8*"-")

    return args

class MRDataset(data.Dataset):
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
        self.dataset_path = f"{self.root_dir}/{subfolder}/{view_type}/"
        self.img_labels = pd.read_csv(f"{self.root_dir}/{subfolder}-abnormal.csv", 
                                      names=["id", "abnormality"], 
                                      dtype={"id": str, "abnormality": int})
        self.img_labels = self.img_labels.set_index("id")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image = np.load(f"{self.dataset_path}/{index}.npy")
        label = self.img_labels.loc[index]["abnormality"]
        
        if self.transform:
            image = self.transform(image)

        return image, label


class MRNet(nn.Module):
    
    def __init__(self, pretrained_model_type):
        super().__init__()
        self.pretrained_model = get_pretrained_model(pretrained_model_type)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        print(f"X before squeeze: {x.size()}")
        x = torch.squeeze(x, dim=0)
        print(f"X after squeeze: {x.size()}")
        features = self.pretrained_model.features(x)
        print(f"features after pretrained_model: {features.size()}")
        pooled_features = self.pooling_layer(features)
        print(f"pooled_features after pooling_layer: {pooled_features.size()}")
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        print(f"pooled_features after view: {pooled_features.size()}")
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        print(f"flattened_features after view: {flattened_features.size()}")
        output = self.classifier(flattened_features)
        return output

def train_model():
    '''
    trains model for recognising selected abnormality on images taken from choosen view
    '''
    pass

if __name__ == "__main__":
    args = get_args()