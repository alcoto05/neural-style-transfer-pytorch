import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatures(nn.Module):
    '''Extracts features from specific layers of VGG19'''
    def __init__(self):
        super(VGGFeatures, self).__init__()
       
        try:
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        except ImportError:
            vgg = models.vgg19(pretrained=True).features

        # Mapping layer indices to names
        self.layer_names = {
            '0': 'conv1_1',      
            '5': 'conv2_1',   
            '10': 'conv3_1',
            '19': 'conv4_1',  
            '21': 'conv4_2',  
            '28': 'conv5_1'  
        }
        self.features = vgg[:29]
        
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        ''' Extracts features from the specified layers'''
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layer_names:
                features[self.layer_names[name]] = x
        return features


def get_vgg_model(device):
    '''returns a pre-trained VGG19 model truncated at certain layers for feature extraction'''
    model = VGGFeatures()
    model.to(device).eval()
    
    print("Model VGGFeatures loaded and moved to device:", device)
    return model