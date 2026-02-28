import torch
import torchvision
from torchvision import transforms as T
import cv2
import os
import numpy as np
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


# Folders
input_folder = "preprocessed_images"
output_folder = "segmented_images"
os.makedirs(output_folder, exist_ok=True)

# Load pre-trained Mask R-CNN

model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Transform for images
transform = T.Compose([T.ToTensor()])

# Process images
for img_file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_file)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # Select masks with score > 0.8
    masks = prediction['masks']
    scores = prediction['scores']
    
    mask_final = torch.zeros_like(masks[0][0])
    for i, score in enumerate(scores):
        if score > 0.8:
            mask_final = torch.maximum(mask_final, masks[i][0])
    
    # Convert mask to numpy
    mask_final = mask_final.mul(255).byte().cpu().numpy()
    
    # Apply mask to original image
    img_np = np.array(img)
    result = cv2.bitwise_and(img_np, img_np, mask=mask_final)
    
    # Save segmented image
    cv2.imwrite(os.path.join(output_folder, img_file), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Segmented {img_file}")