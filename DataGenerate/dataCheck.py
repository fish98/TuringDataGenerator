import os
import numpy as np
from PIL import Image

dirname = "./Data"
dirlist = os.listdir(dirname)
for index, file in enumerate(dirlist):
    bios = 0
    file_path = os.path.join(dirname, file)
    img = Image.open(file_path).convert("L")
    imageArray = np.array(img) / 255.
    firstNum = imageArray[0][0]
    for row in imageArray:
        for col in row:
            bios = bios + abs(col - firstNum)
            
    if(bios < 20000):
        print("NO ---- process {} out of {}".format(index, len(dirlist)))
        os.remove(file_path)
    else:
        print("YES ---- process {} out of {}".format(index, len(dirlist)))
    

