import pandas as pd
import numpy as np
import logging
import torch
from torch import nn
from torch.optim import SGD
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
from models import MriNet, load_checkpoint, save_checkpoint
from transforms import test_transforms, train_transforms

def get_args():
    parser = argparse.ArgumentParser(description='Process paramaters for model learning')
    parser.add_argument('--abnormality_type', type=str, help='abnormal/acl/meniscus')
    parser.add_argument('--root_dir', type=str, help='root_dir/view_type')
    parser.add_argument('--batch_size', type=int, help='Number of images in batch')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')
    parser.add_argument('--load_model', type=str, default="N", help='Y -> continue learning using state_dict, train_history in save_path') 
    parser.add_argument('--model_path', type=str, help='path ot yaml configuration file, save/load model state dictionary')
    args = vars(parser.parse_args())
    
    # directory safe check
    if args["abnormality_type"] not in args["model_path"]:
        logging.warn("Abnormality type not found in model path")
        exit()

    # parse str to boolean
    str_true = ["Y", "y", "Yes", "yes", "true", "True"]
    bool_params = ["load_model"]
    for param in bool_params:
        if args[param] in str_true:
            args[param] = True
        else:
            args[param] = False

    # print input parameters
    logging.info(8*"-")
    logging.info("PARAMETERS")
    logging.info(8*"-")
    for parameter in args.keys():
        logging.info(f"{parameter}: {args[parameter]}")
    logging.info(8*"-")

    return args


def train_model(device, root_dir: str, abnormality_type: str, batch_size: int, 
        n_epochs: int, load_model: bool = False, model_path: str = None):
    '''
    trains model for recognising selected abnormality on images taken from choosen view
    '''

    # initiate model and optimizer
    model = MriNet(model_path, abnormality_type)
    model = model.to(device)
    optimizer = SGD(model.classifier.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    start_epoch = 0

    # set weights if training process should be restarted
    if load_model and model_path is not None:

        # load checkpoint with highest epoch number
        train_history = pd.read_csv(f"{model_path}/train_history.csv", sep="|")
        last_train_epoch = train_history[train_history["epoch"] == train_history["epoch"].max()]
        checkpoint_path = last_train_epoch["checkpoint_path"].iloc[0]

        model, optimizer, last_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch = last_epoch + 1

    for epoch in range(start_epoch, n_epochs):

        # future checkpoint
        checkpoint = {"epoch": epoch, "pretrained_model_type": "main_model"}

        for state, data_transforms in [("train", train_transforms), ("test", test_transforms)]:

            logging.info(f"Epoch {epoch}, State: {state}")

            running_loss = 0.0
            running_corrects = 0

            # dataset, dataloader
            dataset = MainMriDataset(root_dir, state, abnormality_type, transform = data_transforms)
            dataloader = DataLoader(dataset, batch_size, shuffle=True)

            # all datasets have the same length
            len_dataset = len(dataset)

            if state == "train":
                model.train()
            else:
                model.eval()

            for id, batch in enumerate(dataloader, 0):

                with torch.set_grad_enabled(state == 'train'):

                    # progress bar
                    if id % 100 == 0 and id != 0:
                        progress = round(((id + 1) / len_dataset) * 100, 1)
                        progress_loss =  round(running_loss / (id + 1), 2)
                        progress_acc = round(running_corrects / (id + 1), 2)
                        logging.info(f"Progress: {progress}%, loss: {progress_loss}, accuracy: {progress_acc}")
                    
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    # calculate loss
                    outputs = model(images)                    
                    loss = criterion(outputs.float(), labels.float())
                    preds = torch.round(outputs)

                    if state == "train":
                        loss.backward()
                        optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            # save and print epoch statistics
            epoch_loss = round(running_loss / len_dataset, 2)
            epoch_acc = round(running_corrects / len_dataset, 2)
            checkpoint[f"{state}_loss"] = epoch_loss
            checkpoint[f"{state}_acc"] = epoch_acc

            logging.info(f"Epoch: {epoch}, loss: {epoch_loss}, accuracy: {epoch_acc}")

        # save checkpoint
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        save_checkpoint(checkpoint, model_path)

    return model


class MainMriDataset(data.Dataset):
    '''
    Attributes
    ----------
    root_dir : str
        path to root directory. Leads to train and val
    state: str
        train/valid
    abnormality_type: str
        abnormal/acl/meniscus
    transform: 
        set of transformations used for image preprocessing
    '''

    def __init__(self, root_dir, state, abnormality_type, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.state = state
        self.abnormality_type = abnormality_type

        self.subfolder = "train" if state == "train" else "valid"
        self.datasets_path = f"{self.root_dir}/{self.subfolder}"
        self.labels = pd.read_csv(f"{self.root_dir}/{self.subfolder}-{self.abnormality_type}.csv", 
                                      names=["id", "abnormality"], 
                                      dtype={"id": str, "abnormality": int})
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        image_row = self.labels.loc[index]
        label = image_row["abnormality"]
        image_index = image_row["id"]

        image_axial = np.load(f"{self.root_dir}/{self.subfolder}/axial/{image_index}.npy")
        image_coronal = np.load(f"{self.root_dir}/{self.subfolder}/coronal/{image_index}.npy")
        image_sagittal = np.load(f"{self.root_dir}/{self.subfolder}/sagittal/{image_index}.npy")

        if self.transform:
            image_axial = self.transform(image_axial)
            image_coronal = self.transform(image_coronal)
            image_sagittal = self.transform(image_sagittal)

        images = {
                "image_axial": image_axial, 
                "image_coronal": image_coronal, 
                "image_sagittal": image_sagittal
                }

        return images, label


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = get_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(device)

    model = train_model(device, args["root_dir"], args["abnormality_type"], 
                        args["batch_size"], args["n_epochs"], 
                        args["load_model"], args["model_path"])