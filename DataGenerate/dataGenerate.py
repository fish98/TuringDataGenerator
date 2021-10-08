import os
import torch
import numpy as np
from PIL import Image

dirname = "./Data"
nameAll = []
labelAll = []
dataAll = []
fishData = {}

imgSize = 128

for index, file in enumerate(os.listdir(dirname)):
    file_path = os.path.join(dirname, file)
    nameAll.append(file)
    filename = file[2:-4]

    label = filename.split('_')
    tmplabel = []
    for item in label:
        tmplabel.append(float(item))
    labelAll.append(tmplabel)

    img = Image.open(file_path).convert("L")
    img = img.resize((imgSize,imgSize), Image.ANTIALIAS)
    tmpimg = np.array(img)/255.0
    tmpimg = tmpimg[np.newaxis, :]
    imgfile = torch.tensor(tmpimg, dtype=torch.float32)
    dataAll.append(imgfile)

fishData['Data'] = dataAll
fishData['Label'] = labelAll
# print(fishData)
np.save('./newData.npy', fishData)
