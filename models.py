from torchvision import models
from torch import nn
import torch

class SubnetMri(nn.Module):
    '''
    Submodel specialized in recognizing specific abnormality using one of 3 views.
    Build on basis of pretrained model.

    Attributes
    ----------
    pretrained_model_type: str
                           resnet18, resnet34, alexnet
    '''

    def __init__(self, pretrained_model_type: str):
        super().__init__()
        self.pretrained_model_type = pretrained_model_type
        self.pretrained_model = get_pretrained_model(pretrained_model_type)
        self.avg_pooling_layer = nn.AdaptiveAvgPool2d((12, 12))
        self.max_pooling_layer = nn.AdaptiveMaxPool2d((12, 12))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(288, 1)

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
        output = torch.sigmoid(self.classifier(features_concat))
        
        return output


class MriNet(nn.Module):
    '''
    Main model responsible for recognizing specific abnormality.
    Build on basis of 3 submodel each specialized in specific view

    Attributes
    ----------
    subnet_axial: SubnetMri
                  submodel specialized in recognizing specific abnormality using axial view
    subnet_coronal: SubnetMri
                    submodel specialized in recognizing specific abnormality using coronal view
    subnet_sagittal: SubnetMri
                     submodel specialized in recognizing specific abnormality using sagittal view  
    '''

    def __init__(self, subnet_axial: SubnetMri, subnet_coronal: SubnetMri, subnet_sagittal: SubnetMri):
        super().__init__()

        # independent models each responsible for individual view
        self.subnet_axial = subnet_axial
        self.subnet_coronal = subnet_coronal
        self.subnet_sagittal = subnet_sagittal

        # final classification layer
        self.classifier = nn.Linear(3, 1)

    def forward(self, x):

        # output from each of model
        output_subnet_axial = self.subnet_axial(x)
        output_subnet_coronal = self.subnet_coronal(x)
        output_subnet_sagittal = self.subnet_sagittal(x)

        output_concat = torch.cat((output_subnet_axial, output_subnet_coronal, output_subnet_sagittal), dim=0)
        output = torch.sigmoid(self.classifier(output_concat))

        return output


def get_pretrained_model(model_name: str):
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
        model.fc = nn.Identity()

    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Identity()

    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()

    else:
        raise Exception(f"Pretrained model type not found: {model_name}")

    for param in model.parameters():
        param.requires_grad = False

    return model