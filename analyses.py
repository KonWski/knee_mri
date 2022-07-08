import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from models import ViewMriNet, load_checkpoint
from transforms import test_transforms
from view_model_training import ViewDataset
import logging
from torch.nn.functional import softmax

def validate_model(checkpoint_path: str, root_dir: str, device, fill_observation_report: bool):
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

    with torch.no_grad():

        # model, transfer  learning type irrelevant
        model = ViewMriNet(pretrained_model_type, "feature_extraction") 
        optimizer = SGD(model.classifier.parameters(), lr=0.01)
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        model = model.to(device)
        model.eval()

        for state in ["train", "test"]:
            
            logging.info(f"Started validation for {state}")

            # calculated parameters
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
                labels = labels[0].to(device)

                # calculate loss
                outputs = model(images).to(device)
                
                proba = softmax(outputs)                    
                pred = torch.round(proba)

                print(f"proba: {proba}")
                print(f"pred: {pred}")

                # tp, fp, tn, fn
                if torch.all(torch.eq(pred, labels)):
                    if pred.item() == [0, 1]:
                        running_tp += 1
                    else:
                        running_tn += 1
                else:
                    if pred.item() == [1, 0]:
                        running_fp += 1
                    else:
                        running_fn += 1

                if state == "test" and fill_observation_report:
                    ids.append(id)
                    preds.append(pred)

            print(f"running_tp: {running_tp}")
            print(f"running_tn: {running_tn}")
            print(f"running_fp: {running_fp}")
            print(f"running_fn: {running_fn}")

            # statistics
            accuracy = round((running_tp + running_tn) / len_dataset, 2)
            precission = round(running_tp / (running_tp + running_fp), 2) if running_tp + running_fp else 0
            recall = round(running_tp / (running_tp + running_fn), 2) if running_tp + running_fn else 0
            f1_score = round((2 * precission * recall) / (precission + recall), 2) if precission + recall else 0

            stats[f"{state}_accuracy"] = [accuracy]
            stats[f"{state}_precission"] = [precission]
            stats[f"{state}_recall"] = [recall]
            stats[f"{state}_f1_score"] = [f1_score]

        stats["epoch"] = [last_epoch]
        stats = pd.DataFrame(stats)

    # predictions for concrete observations (only for test)
    observations_report = pd.DataFrame({"id": ids, "preds": preds})

    return stats, observations_report