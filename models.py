from torchvision import models
from torch import nn
import torch
import yaml
from torch.optim import Adam
import logging
import pandas as pd
from datetime import datetime
import os
from typing import Dict
from torch.nn.functional import relu

class ViewMriNet(nn.Module):
    '''
    Submodel specialized in recognizing specific abnormality using one of 3 views.
    Build on basis of pretrained model.

    Attributes
    ----------
    pretrained_model_type: str
                           resnet18, resnet34, alexnet
    '''

    def __init__(self, pretrained_model_type: str, transfer_learning_type: str):
        super().__init__()
        self.pretrained_model_type = pretrained_model_type
        self.transfer_learning_type = transfer_learning_type
        self.pretrained_model = get_pretrained_model(pretrained_model_type, self.transfer_learning_type)
        self.avg_pooling_layer = nn.AdaptiveAvgPool2d((6, 6))
        self.max_pooling_layer = nn.AdaptiveMaxPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(72, 2)

    def forward(self, x):

        print(f"Pretrained model input size: {x.size()}")
        x = torch.squeeze(x, dim=0) 
        print(f"Pretrained model input (after squeeze) size: {x.size()}")
        features = self.pretrained_model(x)        
        print(f"Pretrained model output size {features.size()}")
        features = torch.unsqueeze(features, dim=0)
        print(f"features unsqueezed size {features.size()}")

        features_avg = self.avg_pooling_layer(features)
        features_max = self.max_pooling_layer(features)

        features_avg = self.flatten(features_avg)
        features_max = self.flatten(features_max)

        features_avg = torch.squeeze(features_avg, dim=0)
        features_max = torch.squeeze(features_max, dim=0)

        features_concat = torch.cat((features_avg, features_max), dim=0)

        output = self.classifier(features_concat)

        return output


class MainMriNet(nn.Module):
    '''
    Main model responsible for recognizing specific abnormality.
    Build on basis of 3 submodel each specialized in specific view

    Attributes
    ----------

    '''
    def __init__(self, model_path: str, abnormality_type: str, transfer_learning_type: str):
        super().__init__()

        config = self.load_final_model_config(model_path, abnormality_type)

        # load template models        
        self.subnet_axial = ViewMriNet(config["axial"]["pretrained_model_type"], transfer_learning_type)
        self.subnet_coronal = ViewMriNet(config["coronal"]["pretrained_model_type"], transfer_learning_type)
        self.subnet_sagittal = ViewMriNet(config["sagittal"]["pretrained_model_type"], transfer_learning_type)
        
        dummy_optimizer_axial = Adam(self.subnet_axial.parameters(), lr=1e-5)
        dummy_optimizer_coronal = Adam(self.subnet_coronal.parameters(), lr=1e-5)
        dummy_optimizer_sagittal = Adam(self.subnet_sagittal.parameters(), lr=1e-5)

        axial_model_path = config["axial"]["checkpoint_path"]
        subnet_axial, optimizer, last_epoch = load_checkpoint(self.subnet_axial, dummy_optimizer_axial, axial_model_path)
        self.subnet_axial = subnet_axial

        coronal_model_path = config["coronal"]["checkpoint_path"]
        subnet_coronal, optimizer, last_epoch = load_checkpoint(self.subnet_coronal, dummy_optimizer_coronal, coronal_model_path)
        self.subnet_coronal = subnet_coronal

        sagittal_model_path = config["sagittal"]["checkpoint_path"]
        subnet_sagittal, optimizer, last_epoch = load_checkpoint(self.subnet_sagittal, dummy_optimizer_sagittal, sagittal_model_path)
        self.subnet_sagittal = subnet_sagittal
        
        # turn off  grads in all parameters 
        for model in [self.subnet_axial, self.subnet_coronal, self.subnet_sagittal]:
            for param in model.parameters():
                param.requires_grad = False

        # final classification layer
        self.final_classifier = nn.Linear(6, 2)

    def forward(self, image_axial, image_coronal, image_sagittal):

        # output from each of model
        output_subnet_axial = self.subnet_axial(image_axial)
        output_subnet_coronal = self.subnet_coronal(image_coronal)
        output_subnet_sagittal = self.subnet_sagittal(image_sagittal)

        output_concat = torch.cat((output_subnet_axial, output_subnet_coronal, output_subnet_sagittal), dim=0)
        output = self.final_classifier(output_concat)

        return output

    def load_final_model_config(self, model_path, abnormality_type):
        with open(f"{model_path}/{abnormality_type}_config.yml", "r") as config_stream:
            return yaml.safe_load(config_stream)

    
def get_pretrained_model(model_name: str, transfer_learning_type: str):
    '''
    Downloads pretrained model from PyTorch, modifies its layers to output features

    Parameters
    ----------
    model_name: str
        indicator which of PyTorch pretrained model should be returned
        possible options: resnet18, resnet34, alexnet
    '''

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()

    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()

    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
    
    else:
        raise Exception(f"Pretrained model type not found: {model_name}")

    if transfer_learning_type == "feature_extraction":
        for param in model.parameters():
            param.requires_grad = False

    return model


def load_checkpoint(model: nn.Module, optimizer: torch.optim, checkpoint_path: str):
    '''
    loads model checkpoint from given path

    Parameters
    ----------
    model : nn.Module
        One of models defined in pretrained_models scripts
    optimizer : torch.optim
    checkpoint_path : str
        Path to checkpoint

    Notes
    -----
    checkpoint: dict
                parameters retrieved from training process i.e.:
                - model_state_dict
                - optimizer_state_dict
                - last finished number of epoch
                - loss from last epoch training
                - accuracy from last epoch training
                - loss from last epoch testing
                - accuracy from last epoch testing
                - save time
    '''
    checkpoint = torch.load(checkpoint_path)

    # load parameters from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]    

    # print loaded parameters
    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {epoch}")
    logging.info(f"Train loss: {checkpoint['train_loss']}")
    logging.info(f"Train accuracy: {checkpoint['train_acc']}")
    logging.info(f"Test loss: {checkpoint['test_loss']}")
    logging.info(f"Test accuracy: {checkpoint['test_acc']}")
    logging.info(8*"-")

    return model, optimizer, epoch


def save_checkpoint(checkpoint: dict, model_path: str):
    '''
    saves model to checkpoint

    Parameters
    ----------
    checkpoint: dict
            parameters retrieved from training process i.e.:
            - model_state_dict
            - optimizer_state_dict
            - last finished number of epoch
            - loss from last epoch training
            - accuracy from last epoch training
            - loss from last epoch testing
            - accuracy from last epoch testing
            - Save time
    model_path : str
                 Path to directory with checkpoints
    '''
    checkpoint_path = f"{model_path}/{checkpoint['pretrained_model_type']}_{checkpoint['epoch']}"

    # save checkpoint
    # torch.save(checkpoint, checkpoint_path)

    # new row in train history
    new_log = pd.DataFrame({
                            "pretrained_model_type": [checkpoint["pretrained_model_type"]],
                            "epoch": [checkpoint["epoch"]],
                            "train_loss": [checkpoint["train_loss"]],
                            "train_acc": [checkpoint["train_acc"]],
                            "test_loss": [checkpoint["test_loss"]],
                            "test_acc": [checkpoint["test_acc"]],
                            "checkpoint_path": [checkpoint_path],
                            "save_dttm": [datetime.now()]
                            })

    # check if file with training logs already exists
    train_history_path = f"{model_path}/train_history.csv"
    
    if not os.path.exists(train_history_path):
        new_log.to_csv(train_history_path, sep="|", index=False)
    else:
        train_history = pd.read_csv(train_history_path, sep="|")
        train_history = pd.concat([train_history, new_log], ignore_index=True)
        train_history.to_csv(train_history_path, sep="|", index=False)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(8*"-")