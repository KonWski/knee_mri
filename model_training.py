import pandas as pd
import numpy as np
import logging
import torch
from torch import nn
from torch.optim import SGD
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
from models import SubnetMri, MriNet, load_checkpoint, save_checkpoint
import torchvision.transforms as transforms

def get_args():
    parser = argparse.ArgumentParser(description='Process paramaters for model learning')
    parser.add_argument('--view_type', type=str, help='axial/coronal/sagittal')
    parser.add_argument('--abnormality_type', type=str, help='abnormal/acl/meniscus')
    parser.add_argument('--root_dir', type=str, help='root_dir/view_type')
    parser.add_argument('--pretrained_model_type', type=str, help='Type of model used for feature extraction AlexNet/Resnet/Inception')
    parser.add_argument('--batch_size', type=int, help='Number of images in batch')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')
    parser.add_argument('--model_path', type=str, default=None, help='Path to directory to save/load model state dictionary')
    parser.add_argument('--load_model', type=str, default="N", help='Y -> continue learning using state_dict, train_history in save_path')

    args = vars(parser.parse_args())
    
    # directory safe check
    if args["view_type"] not in args["model_path"] or args["abnormality_type"] not in args["model_path"]:
        logging.warn("Abnormality type or view type not found in model path")
        exit()

    # parse str to boolean
    str_true = ["Y", "y", "Yes", "yes", "true", "True"]
    if args["load_model"] in str_true:
        args["load_model"] = True
    else:
        args["load_model"] = False

    # print input parameters
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

    def __init__(self, root_dir, state, view_type, abnormality_type, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.state = state
        self.view_type = view_type
        self.abnormality_type = abnormality_type

        subfolder = "train" if state == "train" else "valid"
        self.dataset_path = f"{self.root_dir}/{subfolder}/{view_type}"
        self.labels = pd.read_csv(f"{self.root_dir}/{subfolder}-{self.abnormality_type}.csv", 
                                      names=["id", "abnormality"], 
                                      dtype={"id": str, "abnormality": int})
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        image_row = self.labels.loc[index]
        label = image_row["abnormality"]
        image_index = image_row["id"]
        image = np.load(f"{self.dataset_path}/{image_index}.npy")

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(device, root_dir: str, view_type: str, abnormality_type: str, pretrained_model_type: str, 
        batch_size: int, n_epochs: int, load_model: bool = False, model_path: str = None):
    '''
    trains model for recognising selected abnormality on images taken from choosen view
    '''

    # basic transformations + augmentation
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
        transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
        ])

    # basic transformations
    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.unsqueeze(x, dim=0)),
            transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
            ])

    # initiate model and optimizer
    model = SubnetMri(pretrained_model_type)
    model = model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"param: {name}")

    optimizer = SGD(model.classifier.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    start_epoch = 0

    # set weights if training process should be restarted
    if load_model and model_path is not None:
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, model_path)
        start_epoch = last_epoch + 1

    for epoch in range(start_epoch, n_epochs):

        # future checkpoint
        checkpoint = {"epoch": epoch, "pretrained_model_type": pretrained_model_type}

        for state, data_transforms in [("train", train_transforms), ("test", test_transforms)]:

            logging.info(f"Epoch {epoch}, State: {state}")

            running_loss = 0.0
            running_corrects = 0

            dataset = MriDataset(root_dir, state, view_type, abnormality_type, transform = data_transforms)
            len_dataset = len(dataset)
            dataloader = DataLoader(dataset, batch_size, shuffle=True)

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


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = train_model(device, args["root_dir"], args["view_type"], args["abnormality_type"], 
                            args["pretrained_model_type"], args["batch_size"], args["n_epochs"], 
                            args["load_model"], args["model_path"])