import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as TorchData
from optuna.trial import TrialState
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import optuna
from optuna import Trial
import os

# Setup
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(message)s')

# Constants
epochsNum = 20
inputDim = 128 * 128
# latentDim = 16
batchSize = 60


# Data
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


# Model
class VAE(nn.Module):
    def __init__(self, latent_dim: int, regress_layer_cnt: int, reg_fc_dim: int):
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
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc31 = nn.Linear(128, 512)
        self.fc32 = nn.Linear(512, 2048)
        self.fc4 = nn.Linear(2048, 16 * 40 * 40)

        self.convertZ = nn.Sequential(
            nn.Linear(latent_dim, reg_fc_dim),
            *([nn.Linear(reg_fc_dim, reg_fc_dim)] * regress_layer_cnt),
            nn.Linear(reg_fc_dim, 4),
        )

    def encode(self, x):
        x1 = torch.reshape(x, (-1, 1, 128, 128))
        tmp1 = F.relu(self.conv1(x1))
        tmp2 = F.relu(self.conv2(tmp1))
        tmp3, indice = self.maxPool(tmp2)
        self.indice = indice
        tmp3 = F.relu(tmp3)
        x2 = torch.reshape(tmp3, (-1, 16 * 40 * 40))
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
        x3 = torch.reshape(tmp10, (-1, 16, 40, 40))
        tmp11 = F.relu(self.maxUnPool(x3, self.indice))
        tmp12 = F.relu(self.reconv1(tmp11))
        out = torch.reshape(torch.sigmoid(self.reconv2(tmp12)), (-1, 128 * 128))
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


# Loss
def cal_loss(genImg, img, mu, logvar, label, preLabel, epoch, bp):
    ####################################
    # genImg: generating images
    # Img: origin images
    # mu: latent mean
    # logvar: latent log variance
    ####################################
    reconstruction_function = nn.MSELoss(size_average=False)
    BCE = reconstruction_function(genImg, img)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLDElement = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLDElement).mul_(- 0.5)
    # KL divergence
    # Label loss
    SmoothL = torch.tensor(F.smooth_l1_loss(label, preLabel, reduction='sum'), dtype=torch.double)

    return BCE + KLD + SmoothL * (10 if epoch < epochsNum/2 else bp)


# Data Loading // All 23735 Train 23400
fishData = np.load("newData.npy", allow_pickle=True).item()
data = np.array(fishData['Data'])
label = np.array(fishData['Label'])

dataset = fishDataset(data, label)
tmptrain, tmpvalid = TorchData.random_split(dataset, [22800, 935])
trainData = TorchData.DataLoader(tmptrain, batch_size=batchSize, num_workers=8)
validData = TorchData.DataLoader(tmpvalid, batch_size=batchSize, num_workers=8)


def objective(trial: Trial):
    # Parameters
    learning_rate = trial.suggest_float('lr', 5e-4, 5e-2, log=True)  # 0.0001
    latent_dim = trial.suggest_int('latent_dim', 8, 32, log=True)  # 16
    regress_layer_cnt = trial.suggest_int('regress_layer_count', 2, 6)  # 2
    reg_fc_layer_dim = trial.suggest_int('reg_fc_layer_dim', 8, 32, log=True)  # 20

    biosPara = trial.suggest_int('bp', 10, 30, log=True)

    # Training
    model = VAE(latent_dim, regress_layer_cnt, reg_fc_layer_dim)
    writer = SummaryWriter()

    if torch.cuda.is_available():
        logging.info('cuda is OK!')
        model = model.to('cuda')
    else:
        logging.warning('cuda is unavailable!')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    global_step = 0
    val_loss = 0

    # TODO 3
    AverageData = torch.zeros(1,1,128,128)
    AverageNumber = 0
    for batch_idx, data in enumerate(tqdm(trainData)):
        img, label = data
        for index, tmpdata in enumerate(range(img.shape[0])):
            AverageData += img[index]
            AverageNumber += 1
    for batch_idx, data in enumerate(tqdm(validData)):
        img, label = data
        for index, tmpdata in enumerate(range(img.shape[0])):
            AverageData += img[index]
            AverageNumber += 1
    AverageData = AverageData / AverageNumber
    AverageData = AverageData.view(AverageData.size(0), -1)

    for epoch in range(epochsNum):
        logging.info(f'Executing {epoch}/{epochsNum} epoch')
        epoch_loss = 0

        for batch_idx, data in enumerate(tqdm(trainData)):
            global_step += 1
            img, label = data
            img = img.view(img.size(0), -1)
            img = img - AverageData
            img = (img.cuda() if torch.cuda.is_available() else img)
            label = (label.cuda() if torch.cuda.is_available() else label)

            optimizer.zero_grad()

            genImg, mu, log_var, preLabel = model(img)

            loss = cal_loss(genImg, img, mu, log_var, label, preLabel, epoch, bp=biosPara)
            writer.add_scalar('vae_loss', loss, global_step)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

            lossFunc = nn.MSELoss(reduction='mean')
            writer.add_scalar('label_loss', lossFunc(label, preLabel), global_step)

            # if batch_idx == 0:
            #     input_images = img.reshape((-1, 1, 128, 128)).repeat_interleave(3, dim=1)
            #     output_images = genImg.reshape((-1, 1, 128, 128)).repeat_interleave(3, dim=1)
            #     writer.add_images('inputs', input_images, global_step)
            #     writer.add_images('outputs', output_images, global_step)
            #     writer.add_images('diff', torch.abs(input_images - output_images), global_step)

        val_loss = 0
        for val_data in tqdm(validData):
            img, label = val_data
            img = img.view(img.size(0), -1)
            img = img - AverageData
            img = (img.cuda() if torch.cuda.is_available() else img)
            label = (label.cuda() if torch.cuda.is_available() else label)
            genImg, mu, log_var, preLabel = model(img)
            lossFunc = nn.MSELoss(reduction='mean')
            val_loss += lossFunc(label, preLabel)
        writer.add_scalar('val_loss', val_loss, global_step)

        trial.report(val_loss, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    writer.add_hparams(
        {'lr': learning_rate,
         'latent_dim': latent_dim,
         'regress_layer_cnt': regress_layer_cnt,
         'reg_fc_layer_dim': reg_fc_layer_dim,
         'bp': biosPara,},
        {'val_loss': val_loss}
    )

    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=80, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
