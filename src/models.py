import torch
import torch.nn as nn
import torchvision.models as models


def get_vgg_model(device):
    '''returns a pre-trained VGG19 model truncated at certain layers for feature extraction'''
    vgg = models.vgg19(pretrained=True)

    # freeze parameters
    for param in vgg.parameters():
        param.requires_grad = False
    
    vgg.to(device)
    print("VGG19 model loaded and moved to device:", device)
    return vgg
