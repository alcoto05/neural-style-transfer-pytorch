import torch
from PIL import Image
import torchvision.transforms as transforms
from src.models import VGGFeatures, get_vgg_model
from src.util import image_loader, imshow, gram_matrix
from torch.optim import Adam
from torchvision.utils import save_image


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def train(content_path, 
    style_path, 
    output_path="outputs/", 
    num_steps=3000, 
    content_weight=1, 
    style_weight=1e6,
    learning_rate=0.003):

    ### LOADING DATA ###
    content_img = image_loader(content_path)
    style_img = image_loader(style_path)

    model=get_vgg_model(device)

    target_content_features = model(content_img)
    target_content_representation = target_content_features['conv4_2'].detach() # Detach to avoid tracking in autograd

    target_style_features = model(style_img)

    style_grams= {}
    for layer, activation in target_style_features.items():
        # GRAM MATRIX
        style_grams[layer] = gram_matrix(activation).detach()

    # Optional: Define weights for each style layer (can be adjusted)
    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2
    }

    ### OPTIMIZER ###
    generated_img = content_img.clone().requires_grad_(True).to(device)
    optimizer=Adam([generated_img],lr=learning_rate)

    ### TRAIN LOOP ###
    for step in range(num_steps):
        
        gen_features = model(generated_img)
        content_loss = torch.mean((gen_features['conv4_2'] - target_content_representation)**2)
        style_loss = 0
        for layer_name, weight in style_weights.items():
            current_feature = gen_features[layer_name]
            
            # Actual gram matrix
            current_gram = gram_matrix(current_feature)
            
            # aim gram matrix
            target_gram = style_grams[layer_name]
            
            # MSE 
            layer_style_loss = torch.mean((current_gram - target_gram)**2)
            style_loss += layer_style_loss * weight

        # Total loss
        total_loss = (content_weight * content_loss) + (style_weight * style_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 10 == 0:  # Every 10 steps we can see the progress
            print(f"Step [{step}/{num_steps}] Loss: {total_loss.item():.2f}")

        # Save image (every 500 steps so we don't flood the output folder)
        if step % 500 == 0:
            save_image(generated_img, f"{output_path}/generated_{step}.png")

if __name__ == "__main__":
    content_file = "inputs/content/puppy-1.jpg"  
    style_file = "inputs/style/the_scream.jpg"     
    
    import os
    if not os.path.exists(content_file) or not os.path.exists(style_file):
        raise FileNotFoundError("Content or style image file not found. Please ensure the paths are correct.")
    else:
        train(content_path=content_file, style_path=style_file)