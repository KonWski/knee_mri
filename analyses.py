import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from models import ViewMriNet, load_checkpoint
from transforms import test_transforms
from view_model_training import ViewDataset
import logging

def validate_model(checkpoint_path: str, root_dir: str, device):
    '''
    - TP, TN, FP TN
    - precission, recall, f1 score
    - info which observation was properly classified
    '''

    # extract from checkpoint_path key infos
    checkpoint_path_split = checkpoint_path.split("/")
    abnormality_type = checkpoint_path_split[-2]
    view_type = checkpoint_path_split[-3]
    pretrained_model_type = checkpoint_path_split[-4]

    # observations, preds, labels
    stats = {
            "abnormality_type": [abnormality_type], 
            "view_type": [view_type], 
            "pretrained_model_type": [pretrained_model_type]
            }
    ids = []
    preds = []
    labels = []

    with torch.no_grad():

        # model
        model = ViewMriNet(pretrained_model_type)
        optimizer = SGD(model.classifier.parameters(), lr=0.01)
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        model = model.to(device)
        model.eval()
        criterion = nn.BCELoss()

        for state in ["train", "test"]:
            
            logging.info(f"Started validation for {state}")

            # calculated parameters
            running_loss = 0.0
            running_tp = 0
            running_fp = 0
            running_tn = 0
            running_fn = 0

            # dataset, dataloader
            dataset = ViewDataset(root_dir, state, view_type, abnormality_type, transform = test_transforms)
            dataloader = DataLoader(dataset, batch_size=1)
            len_dataset = len(dataset)

            for id, batch in enumerate(dataloader, 0):
                
                # progress
                if id % 100 == 0 and id != 0:
                    progress = round(((id + 1) / len_dataset) * 100, 1)
                    logging.info(f"Progress: {progress}%")

                # send images, labels to device
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                # calculate loss
                outputs = model(images)                    
                loss = criterion(outputs.float(), labels.float())
                
                pred = int(torch.round(outputs).item())
                label = labels.item()

                # tp, fp, tn, fn
                if pred == label:
                    if pred == 1:
                        running_tp += 1
                    else:
                        running_tn += 1
                else:
                    if pred == 1:
                        running_fp += 1
                    else:
                        running_fn += 1

                running_loss += loss.item()

                if state == "test":
                    ids.append(id)
                    preds.append(pred)

            # statistics
            loss = round(running_loss / len_dataset, 2)
            accuracy = round((running_tp + running_tn) / len_dataset, 2)
            precission = round(running_tp / (running_tp + running_fp), 2)
            recall = round(running_tp / (running_tp + running_fn), 2)
            f1_score = round((2 * precission * recall) / (precission + recall), 2)

            stats[f"{state}_loss"] = [loss]
            stats[f"{state}_accuracy"] = [accuracy]
            stats[f"{state}_precission"] = [precission]
            stats[f"{state}_recall"] = [recall]
            stats[f"{state}_f1_score"] = [f1_score]

        stats["epoch"] = [last_epoch]
        stats = pd.DataFrame(stats)

    # predictions for concrete observations (only for test)
    observations_report = pd.DataFrame({"id": ids, "preds": preds})

    return stats, observations_report