import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter

from PIL import Image,ImageFile
import albumentations as A
import matplotlib.pyplot as plt

from sklearn import model_selection
import segmentation_models_pytorch as smp

import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import gradio as gr
import tempfile

import io

Encoder = 'resnet34'
Weights = 'imagenet'

model = smp.Unet(encoder_name=Encoder, encoder_weights=None, classes=1) # If no work, download weights manually
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()
print('Model Loaded')

def prepare_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the same preprocessing as during training
    preprocess = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    preprocessed = preprocess(image=image)
    return preprocessed['image'].unsqueeze(0)  # Add batch dimension

def predict_mask(image_path, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        input_tensor = prepare_image(image_path).to(device)
        output = model(input_tensor)

    return output

def post_process(output):
    # Convert to numpy array and squeeze extra dimensions
    mask = output.cpu().numpy().squeeze()

    # Apply threshold to create binary mask
    mask = (mask > 0.5).astype(np.uint8)

    return mask

def visualize_result(image_path, mask):
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize mask to match original image size if necessary
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax1.imshow(image)
    ax1.set_title('Original X-ray image')
    ax1.axis('off')

    # Mask
    ax2.imshow(mask_resized, cmap='binary')
    ax2.set_title('The mask')
    ax2.axis('off')

    # Mask overlay
    ax3.imshow(image)
    ax3.imshow(mask_resized, cmap='binary', alpha=0.3)
    ax3.set_title('Mask on the X-ray image')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


def process_new_image(image):
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "uploaded_image.png")
    
    # Save the uploaded image
    image.save(temp_path)

    # Now you have a file path to work with
    output = predict_mask(temp_path, model)

    # Post-process the output
    mask = post_process(output)

    # Use visualize_result function
    fig = plt.figure(figsize=(15, 5))
    visualize_result(temp_path, mask)
    
    # Save the plot to a byte buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Load the image from the byte buffer
    result_image = Image.open(buf)

    # Clean up: remove the temporary file and close the plot
    os.remove(temp_path)
    plt.close(fig)

    return result_image

def usage(image):
    try:
        print('Usage started')
        result = process_new_image(image)
        return result
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

myInterface = gr.Interface(
    fn=usage,
    inputs=gr.Image(type='pil'),
    outputs=gr.Image(type='pil'),
    live=True,
    description='Pneumothorax Segmentation'
)

myInterface.launch(share=True)