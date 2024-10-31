import cv2
import numpy as np
import os
from glob import glob


root_dir = 'faces'

# Define rotation parameters
rotation_angle = -30  # Rotation angle in degrees
crop_percentage = (abs(rotation_angle) // 10) * 5 / 100.0


for filepath in glob(os.path.join(root_dir, '**', '*.png'), recursive=True):
  
    image = cv2.imread(filepath)
    if image is None:
        print(f"Could not read {filepath}. Skipping...")
        continue

   
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

   
    border_x = int(crop_percentage * w)  # Width-based border
    border_y = int(crop_percentage * h)  # Height-based border

    
    if h <= 2 * border_y or w <= 2 * border_x:
        print(f"Image too small for border cropping: {filepath}")
        continue

    
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

   
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    
    cropped_image = rotated_image[border_y:h-border_y, border_x:w-border_x]

    
    if cropped_image.size == 0:
        print(f"Cropping resulted in an empty image: {filepath}")
        continue

  
    original_folder = os.path.basename(os.path.dirname(filepath))
    output_dir = os.path.join(os.path.dirname(filepath), f'{original_folder}_{rotation_angle}_degrees')
    os.makedirs(output_dir, exist_ok=True)

    
    original_filename = os.path.splitext(os.path.basename(filepath))[0]
    new_filename = f"{original_filename}_{rotation_angle}_degrees.png"
    save_path = os.path.join(output_dir, new_filename)

    
    cv2.imwrite(save_path, cropped_image)
    print(f"Processed and saved: {save_path}")
