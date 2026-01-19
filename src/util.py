import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128 # Depending on whether GPU is available, set image size


loader = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.CenterCrop(imsize), 
    transforms.ToTensor()]) 

def image_loader(image_name):
    """Loads an image and returns it as a tensor to the GPU/CPU."""
    image = Image.open(image_name)

    image = loader(image).unsqueeze(0)
    
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
