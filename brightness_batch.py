import cv2
import os
from glob import glob


root_dir = 'faces'

# Define the brightness change percentage
brightness_percentage = 50  


brightness_factor = 1 + (brightness_percentage / 100.0)


for filepath in glob(os.path.join(root_dir, '**', '*.png'), recursive=True):

    image = cv2.imread(filepath)
    if image is None:
        print(f"Could not read {filepath}. Skipping...")
        continue

  
    brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

  
    original_folder = os.path.basename(os.path.dirname(filepath))
    output_dir = os.path.join(os.path.dirname(filepath), f'{original_folder}_{brightness_percentage}_percent')
    os.makedirs(output_dir, exist_ok=True)

    
    original_filename = os.path.splitext(os.path.basename(filepath))[0]
    new_filename = f"{original_filename}_{brightness_percentage}_percent.png"
    save_path = os.path.join(output_dir, new_filename)

    # Save the brightened image
    cv2.imwrite(save_path, brightened_image)
    print(f"Processed and saved: {save_path}")
