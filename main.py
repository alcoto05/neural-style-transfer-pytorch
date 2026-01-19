# main.py
import os
from src.train import train

CONTENT_FILENAME=input("Introduzca el nombre del archivo de la imagen de contenido (con extensión, por ejemplo, content.jpg) y presione Enter: ")

STYLE_FILENAME = input("Introduzca el nombre del archivo de la imagen de estilo (con extensión, por ejemplo, style.jpg) y presione Enter: ")

# Construimos las rutas completas
content_path = os.path.join("inputs", "content", CONTENT_FILENAME)
style_path = os.path.join("inputs", "style", STYLE_FILENAME)

if not os.path.exists(content_path) or not os.path.exists(style_path):
    raise FileNotFoundError("Content or style image file not found. Please ensure the paths are correct.")
else:
    train(
        content_path=content_path,
        style_path=style_path,
        num_steps=3000,  
        output_path="outputs",
        learning_rate=0.003
    )


