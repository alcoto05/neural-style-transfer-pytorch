import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128 # Depending on whether GPU is available, set image size


loader = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.CenterCrop(imsize), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

def image_loader(image_name):
    """Loads an image and returns it as a tensor to the GPU/CPU."""
    image = Image.open(image_name)

    image = loader(image).unsqueeze(0) # Add batch dimension
    
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    """Displays a tensor as an image."""
    unloader = transforms.ToPILImage()  
    image = tensor.cpu().clone()  
    image = image.squeeze(0)      
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def gram_matrix(tensor):
    """Calculates the Gram matrix of a given tensor."""
    b, c, h, w = tensor.size() #batch, channels, height, width 
    features = tensor.view(b * c, h * w)  # reshaping
    G = torch.mm(features, features.t())  
    return G.div(b * c * h * w) # normalize by total number of elements

def get_denormalize_transform(device):
    """Creates a denormalization transform for images based on ImageNet stats."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def denormalize(tensor):
        tensor = tensor * std + mean
        return torch.clamp(tensor, 0, 1) 
        
    return denormalize