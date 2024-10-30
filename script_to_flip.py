from PIL import Image
import os

# Set the path to the folder containing the images
person_name = "tammo"
folder_path = "faces/" + person_name

# Create a new folder for the augmented dataset
augmented_folder = os.path.join(folder_path, person_name + "_flipped")
os.makedirs(augmented_folder, exist_ok=True)

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Skip files with a .bad extension
    if filename.lower().endswith('.bad'):
        print(f"Skipping file with .bad extension: {filename}")
        continue

    try:
        # Open the image file
        with Image.open(file_path) as img:

            # Flip image across x = 0 (horizontal flip)
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Rename "left" to "right" and vice versa in the filename
            base_name, ext = os.path.splitext(filename)
            if "left" in base_name.lower():
                flipped_base_name = base_name.lower().replace("left", "right")
            elif "right" in base_name.lower():
                flipped_base_name = base_name.lower().replace("right", "left")
            else:
                flipped_base_name = base_name

            # Append unique identifier for flipped image
            flipped_filename = f"{flipped_base_name}_flipped_x0{ext}"
            flipped_img.save(os.path.join(augmented_folder, flipped_filename))

            print(f"Original image saved as {filename}, flipped image saved as {flipped_filename}")

    except IOError:
        print(f"Skipping file {filename}. It is not a valid image.")

