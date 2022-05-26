from torchvision import models
from torch import nn

def get_pretrained_model(model_name: str):
    '''
    Downloads pretrained model from PyTorch, changes its last layer
    and returns it.

    Parameters
    ----------
    model_name: str
        indicator which of PyTorch pretrained model should be returned
    '''

    if model_name:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Identity()

        for param in model.parameters():
            param.requires_grad = False
    else:
        raise Exception(f"Pretrained model type not found: {model_name}")

    return model