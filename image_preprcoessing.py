import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# Load the SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)

# Function to segment an image
def segment_window(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict()
    
    segmented_image = np.zeros_like(image_rgb)
    segmented_image[masks[0]] = image_rgb[masks[0]]

    return segmented_image

# Example usage
segmented_window = segment_window("window.jpeg")
cv2.imwrite("segmented_window.jpg", segmented_window)
