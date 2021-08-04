import os
import datetime
import numpy as np
import torch
from torch import nn
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as torchdata
from torch.utils.data import DataLoader
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def toImg(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 64, 64)
    return x

# Parameters

epochsNum = 30000
batchSize = 32
learningRate = 0.0005

inputDim = 64*64
latentDim = 20

# Data Loading
dirname = "./Data"
dirlist = os.listdir(dirname)
dataAll = []
for index, file in enumerate(dirlist):
    file_path = os.path.join(dirname, file)
    img = Image.open(file_path).convert("L")
    img = img.resize((64,64), Image.ANTIALIAS)
    tmpimg = np.array(img)/255.0
    tmpimg = tmpimg[np.newaxis, :]
    testImg = torch.tensor(tmpimg, dtype=torch.float32)
    dataAll.append(testImg)

class fishDataset(torchdata.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)

# Model 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(inputDim, 800)
        self.fc11 = nn.Linear(800, 200)
        self.fc21 = nn.Linear(200, latentDim)
        self.fc22 = nn.Linear(200, latentDim)
        self.fc3 = nn.Linear(latentDim, 200)
        self.fc31 = nn.Linear(200, 800)
        self.fc4 = nn.Linear(800, inputDim)
        self.convertZ = nn.Linear(latentDim, 4)

    def encode(self, x):
        tmp1 = F.relu(self.fc1(x))
        tmp2 = F.relu(self.fc11(tmp1))
        return self.fc21(tmp2), self.fc22(tmp2)

    def deParametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, latent):
        tmp3 = F.relu(self.fc3(latent))
        tmp4 = F.relu(self.fc31(tmp3))
        return torch.sigmoid(self.fc4(tmp4))

    def convertLantent(self, latent):
        labels = self.convertZ(latent)
        labels = Variable(labels)
        return labels

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = self.deParametrize(mu, logvar)
        preLabel = self.convertLantent(latent)
        # print(preLabel)
        return self.decode(latent), mu, logvar, preLabel

# Training 

strattime = datetime.datetime.now()
model = VAE()

if torch.cuda.is_available():
    # model.cuda()
    print('cuda is OK!')
    model = model.to('cuda')
else:
    print('cuda is unavailable!')

# reconstruction_function = nn.MSELoss(size_average=False)

# # Save model
# torch.save(model.state_dict(), './fishvae.pth')
model.load_state_dict(torch.load('./fishvae.pth'))

# for batch_idx, data in enumerate(Data):
# print(testImg.shape)
for index, img in enumerate(dataAll):
    
    img = img.view(img.size(0), -1)
    img = Variable(img)
    img = (img.cuda() if torch.cuda.is_available() else img)

    genImg, mu, logvar, preLabel = model(img)

    print("the {}th Image -- Predict as {}".format(index, preLabel))