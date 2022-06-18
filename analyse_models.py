import pandas as pd
import numpy as np
import logging
import torch
from torch import nn
from torch.optim import SGD
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
from models import ViewMriNet, load_checkpoint
from transforms import test_transforms, train_transforms
from view_model_training import ViewDataset

def validate_model(checkpoint_path: str, root_dir: str, device):
    '''
    - TP, TN, FP TN
    - precission, recall, f1 score
    - info which observation was properly classified
    '''

    report = ""

    # extract from checkpoint_path key infos
    checkpoint_path_split = checkpoint_path.split("/")
    abnormality_type = checkpoint_path_split[-2]
    view_type = checkpoint_path_split[-3]
    pretrained_model_type = checkpoint_path_split[-4]

    for state in ["train", "test"]:
        
        # calculated parameters
        running_loss = 0.0
        running_corrects = 0
        running_tp = 0
        running_fp = 0
        running_tn = 0
        running_fn = 0

        # dataset, dataloader
        dataset = ViewDataset(root_dir, state, view_type, abnormality_type, transform = test_transforms)
        dataloader = DataLoader(dataset, batch_size=1)
        len_dataset = len(dataset)

        # model
        model = ViewMriNet(pretrained_model_type)
        optimizer = SGD(model.classifier.parameters(), lr=0.01)
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        criterion = nn.BCELoss()
        model.eval()

        for id, batch in enumerate(dataloader, 0):
            
            # send images, labels to device
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # calculate loss
            outputs = model(images)                    
            loss = criterion(outputs.float(), labels.float())
            preds = torch.round(outputs)

            print(f"outputs: {outputs}")
            print(f"preds: {preds}")

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()

    # save and print epoch statistics
    epoch_loss = round(running_loss / len_dataset, 2)
    epoch_acc = round(running_corrects / len_dataset, 2)

