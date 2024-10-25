# Description: This script converts all .pgm files in a directory to .png files.

import os
import cv2


#change name of directory to the directory where the images are stored
directory = 'faces/tammo/'


for file in os.listdir(directory):
    filename, extension = os.path.splitext(file)
    if extension == ".pgm":
       
        new_file = "{}.png".format(filename)
        
       
        img = cv2.imread(os.path.join(directory, file), cv2.IMREAD_UNCHANGED)
        
       
        cv2.imwrite(os.path.join(directory, new_file), img)
        
       
        os.remove(os.path.join(directory, file))
