import cv2
import numpy as np
import os

input_folder = "captured_images"
output_folder = "preprocessed_images"
os.makedirs(output_folder, exist_ok=True)

for img_file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_file)
    img = cv2.imread(img_path)
    
    # Resize
    img_resized = cv2.resize(img, (512, 512))
    
    # Optional: simple background removal using threshold
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    result = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
    # Save processed image
    cv2.imwrite(os.path.join(output_folder, img_file), result)
    print(f"Processed {img_file}")