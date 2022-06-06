from torchvision import models
from torch import nn

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
        param.requires_grad = True

    return model