import os
import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as torchdata
from torch.utils.data import DataLoader
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if not os.path.exists('./fish_img'):
    os.mkdir('./fish_img')

def toImg(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 128, 128)
    return x

# Parameters

epochsNum = 30000
batchSize = 64
learningRate = 0.0005

inputDim = 128*128
latentDim = 20

# Data Loading

fishData = np.load("fishData128.npy", allow_pickle=True).item()
data = np.array(fishData['Data'])
label = np.array(fishData['Label'])

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

dataset = fishDataset(data, label)
tmptrain, tmpvalid = torchdata.random_split(dataset, [3000, 775])
trainData = torchdata.DataLoader(tmptrain, batch_size=batchSize, num_workers=24)
validData = torchdata.DataLoader(tmpvalid, batch_size=batchSize, num_workers=24)

# Model 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # self.fc1 = nn.Linear(inputDim, 800)
        # self.fc11 = nn.Linear(800, 200)
        # self.fc21 = nn.Linear(200, latentDim)
        # self.fc22 = nn.Linear(200, latentDim)
        # self.fc3 = nn.Linear(latentDim, 200)
        # self.fc31 = nn.Linear(200, 800)
        # self.fc4 = nn.Linear(800, inputDim)
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

reconstruction_function = nn.MSELoss(size_average=False)

def calLoss(genImg, img, mu, logvar, label, preLabel):
    ####################################
    # genImg: generating images
    # Img: origin images
    # mu: latent mean
    # logvar: latent log variance
    ####################################
    BCE = reconstruction_function(genImg, img)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLDElement = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLDElement).mul_(- 0.5)
    # KL divergence
    # Label loss
    SmoothL = torch.tensor(F.smooth_l1_loss(label, preLabel, reduction='sum'), dtype=torch.double)

    return BCE + KLD + SmoothL * 10

optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(epochsNum):
    model.train()
    trainLoss = 0
    
    for batch_idx, data in enumerate(trainData):    
        img, label = data
        # print(label)
        img = img.view(img.size(0), -1)
        img = Variable(img)
        img = (img.cuda() if torch.cuda.is_available() else img)
        label = Variable(label)
        label = (label.cuda() if torch.cuda.is_available() else label)
        
        optimizer.zero_grad()
        
        genImg, mu, logvar, preLabel = model(img)
        
        loss = calLoss(genImg, img, mu, logvar, label, preLabel)
        loss.backward()
        trainLoss += loss.item()
        optimizer.step()

        if batch_idx % 50 == 0:
            endtime = datetime.datetime.now()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
                epoch,
                batch_idx * len(img),
                len(trainData.dataset), 
                100. * batch_idx / len(trainData),
                loss.item() / len(img), 
                (endtime-strattime).seconds))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, trainLoss / len(trainData.dataset)))
    
    if epoch % 1000 == 0:
        save1 = toImg(img)
        save2 = toImg(genImg.cpu().data)
        save_image(save1, './fish_img/image_{}.png'.format(epoch))
        save_image(save2, './fish_img/original_image_{}.png'.format(epoch))

    if epoch != 0 and epoch % 1000 == 0:
        lossFunc = nn.MSELoss(reduction='mean')
        print('Training Accuracy: {}'.format(lossFunc(label, preLabel)))

# Save model
torch.save(model.state_dict(), './fishvae.pth')
