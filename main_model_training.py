import pandas as pd
import numpy as np
import logging
import torch
from torch import nn
from torch.optim import Adam
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
from models import MainMriNet, load_checkpoint, save_checkpoint
from transforms import test_transforms, train_transforms
from torch.nn.functional import softmax
from os.path import exists

def get_args():
    parser = argparse.ArgumentParser(description='Process paramaters for model learning')
    parser.add_argument('--abnormality_type', type=str, help='abnormal/acl/meniscus')
    parser.add_argument('--root_dir', type=str, help='root_dir/view_type')
    parser.add_argument('--transfer_learning_type', type=str, default="feature_extraction", help='feature_extraction/fine_tunning')
    parser.add_argument('--batch_size', type=int, help='Number of images in batch')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')
    parser.add_argument('--load_model', type=str, default="N", help='Y -> continue learning using state_dict, train_history in save_path')
    parser.add_argument('--use_weights', type=str, help='weight observations in loss function, weights calculated automatically')
    parser.add_argument('--model_path', type=str, help='path to yaml configuration file, save/load model state dictionary')
    args = vars(parser.parse_args())
    
    # directory safe check
    if args["abnormality_type"] not in args["model_path"]:
        logging.warn("Abnormality type not found in model path")
        exit()

    # parse str to boolean
    str_true = ["Y", "y", "Yes", "yes", "true", "True"]
    bool_params = ["load_model", "use_weights"]
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


def train_model(device, root_dir: str, abnormality_type: str,  transfer_learning_type: str, batch_size: int, 
        n_epochs: int, use_weights: bool = False,  load_model: bool = False, model_path: str = None):
    '''
    trains model for recognising selected abnormality on images taken from choosen view
    '''

    train_history_path = f"{model_path}/train_history.csv"

    # continue learning from last checkpoint
    if load_model and exists(train_history_path):

        # load checkpoint with highest epoch number
        train_history = pd.read_csv(f"{model_path}/train_history.csv", sep="|")
        last_epoch = train_history["epoch"].max()
        last_training_history = train_history[train_history["epoch"] == train_history["epoch"].max()]
        checkpoint_path = last_training_history["checkpoint_path"].iloc[0]

        # check if training already finished
        if n_epochs <= last_epoch + 1:
            logging.info("Model already trained for given number of epochs")
            exit()

    elif load_model and not exists(train_history_path):
        logging.warning("Further model training impossible - no saved checkpoints")
        exit()

    # check if start training is possible
    elif not load_model and exists(train_history_path):
        logging.warning("""Model already has saved checkpoints. If You wish to rerun whole 
                           training, delete manually existing files (checkpoints, train_history) and start over""")
        exit()

    # initiate model and optimizer
    model = MainMriNet(model_path, abnormality_type, transfer_learning_type)
    model = model.to(device)
    optimizer = Adam(model.final_classifier.parameters(), lr=1e-5)
    start_epoch = 0

    # set weights if training process should be restarted
    if load_model:
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch = last_epoch + 1

    for epoch in range(start_epoch, n_epochs):

        # future checkpoint
        checkpoint = {"epoch": epoch, "pretrained_model_type": "main_model"}

        for state, data_transforms in [("train", train_transforms), ("test", test_transforms)]:

            logging.info(f"Epoch {epoch}, State: {state}")

            # calculated parameters
            running_loss = 0.0
            running_corrects = 0

            dataset = MainDataset(root_dir, state, abnormality_type, use_weights, transform = data_transforms)
            dataloader = DataLoader(dataset, batch_size, shuffle=True)
            len_dataset = len(dataset)

            if use_weights:
                weights = dataset.weights.to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
            else:
                criterion = nn.BCEWithLogitsLoss()

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
                    
                    image_axial, image_coronal, image_sagittal, labels = batch
                    image_axial = image_axial.to(device)
                    image_coronal = image_coronal.to(device)
                    image_sagittal = image_sagittal.to(device)
                    labels = labels[0].to(device)
                    optimizer.zero_grad()

                    # calculate loss
                    outputs = model(image_axial, image_coronal, image_sagittal).to(device)
                    loss = criterion(outputs.float(), labels.float())
                    proba = softmax(outputs)
                    preds = torch.round(proba)

                    if state == "train":
                        loss.backward()
                        optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_corrects += torch.sum(torch.argmax(preds) == torch.argmax(labels)).item()

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


class MainDataset(data.Dataset):
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

    def __init__(self, root_dir, state, abnormality_type, use_weights=False, transform=None):
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
        self.use_weights = use_weights
        if self.use_weights:
            self.weights = self._get_weights()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        image_row = self.labels.loc[index]
        label = image_row["abnormality"]
        image_index = image_row["id"]

        image_axial = np.load(f"{self.root_dir}/{self.subfolder}/axial/{image_index}.npy")
        image_coronal = np.load(f"{self.root_dir}/{self.subfolder}/coronal/{image_index}.npy")
        image_sagittal = np.load(f"{self.root_dir}/{self.subfolder}/sagittal/{image_index}.npy")

        # label encoding
        if label == 1:
            label = torch.tensor([0, 1])
        elif label == 0:
            label = torch.tensor([1, 0])

        if self.transform:
            image_axial = self.transform(image_axial)
            image_coronal = self.transform(image_coronal)
            image_sagittal = self.transform(image_sagittal)

        return image_axial, image_coronal, image_sagittal, label
    
    def _get_weights(self):
        '''
        calculates pos weight for each class according to the suggestion
        given in: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        '''
        pos = len(self.labels[self.labels["abnormality"] == 1])
        neg = len(self.labels[self.labels["abnormality"] == 0])

        pos_weight = neg / pos
        return torch.tensor([1, pos_weight])


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = get_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    model = train_model(device, args["root_dir"], args["abnormality_type"], 
                        args["transfer_learning_type"], args["batch_size"], 
                        args["n_epochs"], args["use_weights"], args["load_model"], 
                        args["model_path"])