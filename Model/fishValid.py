import os
import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as TorchData
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

epochsNum = 400
batchSize = 60
learningRate = 0.0003

inputDim = 128 * 128
latentDim = 5

# Data Loading

fishData = np.load("fishData128.npy", allow_pickle=True).item()
data = np.array(fishData['Data'])
label = np.array(fishData['Label'])

class fishDataset(TorchData.Dataset):
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
tmptrain, tmpvalid = TorchData.random_split(dataset, [3000, 775])
trainData = TorchData.DataLoader(tmptrain, batch_size=batchSize, num_workers=24)
validData = TorchData.DataLoader(tmpvalid, batch_size=batchSize, num_workers=24)

# Model 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.indice = 0
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)

        self.maxPool = nn.MaxPool2d(3, 3, return_indices=True)
        self.maxUnPool = nn.MaxUnpool2d(3, 3)   

        self.reconv1 = nn.ConvTranspose2d(16, 8, 5)
        self.reconv2 = nn.ConvTranspose2d(8, 1, 5)

        self.fc1 = nn.Linear(16 * 40 * 40, 2048)
        self.fc11 = nn.Linear(2048, 512)
        self.fc12 = nn.Linear(512, 128)
        self.fc21 = nn.Linear(128, latentDim)
        self.fc22 = nn.Linear(128, latentDim)

        self.fc3 = nn.Linear(latentDim, 128)
        self.fc31 = nn.Linear(128, 512)
        self.fc32 = nn.Linear(512, 2048)
        self.fc4 = nn.Linear(2048, 16 * 40 * 40)

        # self.convertZ = nn.Linear(latentDim, 4)
        self.convertZ = nn.Sequential(
            nn.Linear(latentDim, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 4),
        )

    def encode(self, x):
        x1 = torch.reshape(x, (batchSize, 1, 128, 128))
        tmp1 = F.relu(self.conv1(x1))
        tmp2 = F.relu(self.conv2(tmp1))
        tmp3, indice = self.maxPool(tmp2)
        self.indice = indice
        tmp3 = F.relu(tmp3)
        x2 = torch.reshape(tmp3, (batchSize, 16 * 40 * 40))
        tmp4 = F.relu(self.fc1(x2))
        tmp5 = F.relu(self.fc11(tmp4))
        tmp6 = F.relu(self.fc12(tmp5))
        return self.fc21(tmp6), self.fc22(tmp6)

    def deparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, latent):
        tmp7 = F.relu(self.fc3(latent))
        tmp8 = F.relu(self.fc31(tmp7))
        tmp9 = F.relu(self.fc32(tmp8))
        tmp10 = F.relu(self.fc4(tmp9))
        x3 = torch.reshape(tmp10, (batchSize, 16, 40, 40))
        tmp11 = F.relu(self.maxUnPool(x3, self.indice))
        tmp12 = F.relu(self.reconv1(tmp11))
        out = torch.reshape(torch.sigmoid(self.reconv2(tmp12)), (batchSize, 128 * 128))
        return out

    def convert_lantent(self, latent):
        labels = self.convertZ(latent)
        labels = Variable(labels)
        return labels

    def forward(self, x):
        mu, log_var = self.encode(x)
        latent = self.deparametrize(mu, log_var)
        pre_label = self.convert_lantent(latent)
        return self.decode(latent), mu, log_var, pre_label

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

# def calLoss(genImg, img, mu, logvar, label, preLabel):
#     ####################################
#     # genImg: generating images
#     # Img: origin images
#     # mu: latent mean
#     # logvar: latent log variance
#     ####################################
#     BCE = reconstruction_function(genImg, img)  # mse loss
#     # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLDElement = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#     KLD = torch.sum(KLDElement).mul_(- 0.5)
#     # KL divergence
#     # Label loss
#     SmoothL = torch.tensor(F.smooth_l1_loss(label, preLabel, reduction='sum'), dtype=torch.double)

#     return BCE + KLD + SmoothL * 10

# optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# for epoch in range(epochsNum):
#     model.train()
#     trainLoss = 0
    
#     for batch_idx, data in enumerate(trainData):    
#         img, label = data
#         # print(label)
#         img = img.view(img.size(0), -1)
#         img = Variable(img)
#         img = (img.cuda() if torch.cuda.is_available() else img)
#         label = Variable(label)
#         label = (label.cuda() if torch.cuda.is_available() else label)
        
#         optimizer.zero_grad()
        
#         genImg, mu, logvar, preLabel = model(img)
        
#         loss = calLoss(genImg, img, mu, logvar, label, preLabel)
#         loss.backward()
#         trainLoss += loss.item()
#         optimizer.step()

#         if batch_idx % 50 == 0:
#             endtime = datetime.datetime.now()
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
#                 epoch,
#                 batch_idx * len(img),
#                 len(trainData.dataset), 
#                 100. * batch_idx / len(trainData),
#                 loss.item() / len(img), 
#                 (endtime-strattime).seconds))
#     print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, trainLoss / len(trainData.dataset)))
    
#     if epoch % 1000 == 0:
#         save1 = toImg(img)
#         save2 = toImg(genImg.cpu().data)
#         save_image(save1, './fish_img/image_{}.png'.format(epoch))
#         save_image(save2, './fish_img/original_image_{}.png'.format(epoch))

#     if epoch != 0 and epoch % 1000 == 0:
#         lossFunc = nn.MSELoss(reduction='mean')
#         print('Training Accuracy: {}'.format(lossFunc(label, preLabel)))

# # Save model
# torch.save(model.state_dict(), './fishvae.pth')
model.load_state_dict(torch.load('./fishvae.pth'))
for batch_idx, data in enumerate(validData):
        errorV = 0
        img, label = data
        # print(label)
        img = img.view(img.size(0), -1)
        img = Variable(img)
        img = (img.cuda() if torch.cuda.is_available() else img)
        label = Variable(label)
        label = (label.cuda() if torch.cuda.is_available() else label)
        
        genImg, mu, logvar, preLabel = model(img)
        
        lossFunc = nn.MSELoss(reduction='mean')
        errorV = lossFunc(label, preLabel)

        print(label)
        print(preLabel)
        print("The MSE Loss is {}".format(errorV))
