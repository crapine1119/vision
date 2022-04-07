import pandas as pd
import numpy as np
import sys
import glob
import os
import random
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as mms
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,Dataset
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import cv2 as cv
from glob import glob as glob
from tqdm import tqdm
from torchvision import transforms
from PIL import Image as pil
from gan.modules import *
from torchvision.utils import make_grid
##
fdir = r'D:\cv\Dataset\mpii_human_pose_v1.tar\images'
train = glob(f'{fdir}/*')[:1000]

trnset = custom(train)
b_size = 256
trn_loader = DataLoader(trnset, batch_size=b_size, drop_last=True, shuffle=True, num_workers=2, pin_memory=True)

real_batch = next(iter(trn_loader))
##

fig = plt.figure(figsize = [18,8])
plt.imshow(make_grid(real_batch, padding=2, normalize=True).permute(1,2,0))
##
nz = 100 # generator input
ngf = 64 # G feature
ndf = 64 # D feature
nc = 3

netG = Generator()
netG.apply(weights_init)
netG = netG.cuda()

netD = Discriminator()
netD.apply(weights_init)
netD = netD.cuda()
##
lr = 0.0002
num_epochs = 200

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1).cuda()
real_label = 1.
fake_label = 0.

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5,0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5,0.999))
##
img_list = []
G_losses = []
D_losses = []
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(tqdm(trn_loader), 0):
        # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
        # 진짜 데이터들로 학습을 합니다
        netD.zero_grad()
        real_cpu = data.cuda()
        label = torch.full((b_size,), real_label, dtype=torch.float).cuda()
        label.fill_(real_label)  # fake labels are real for generator cost
        # 진짜 데이터들로 이루어진 배치를 D에 통과시킵니다
        output = netD(real_cpu).view(-1)
        # 손실값을 구합니다
        errD_real = criterion(output, label)
        # 역전파의 과정에서 변화도를 계산합니다
        errD_real.backward()
        D_x = output.mean().item()

        # 가짜 데이터들로 학습을 합니다
        # 생성자에 사용할 잠재공간 벡터를 생성합니다
        noise = torch.randn(b_size, nz, 1, 1).cuda()
        # G를 이용해 가짜 이미지를 생성합니다
        fake = netG(noise)
        label.fill_(fake_label)
        # D를 이용해 데이터의 진위를 판별합니다
        output = netD(fake.detach()).view(-1)
        # D의 손실값을 계산합니다
        errD_fake = criterion(output, label)
        # 역전파를 통해 변화도를 계산합니다. 이때 앞서 구한 변화도에 더합니다(accumulate)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값들을 더합니다
        # 이때 errD는 역전파에서 사용되지 않고, 이후 학습 상태를 리포팅(reporting)할 때 사용합니다
        errD = errD_real + errD_fake
        # D를 업데이트 합니다
        optimizerD.step()

        # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
        netG.zero_grad()
        label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
        # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
        # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
        output = netD(fake).view(-1)
        # G의 손실값을 구합니다
        errG = criterion(output, label)
        # G의 변화도를 계산합니다
        errG.backward()
        D_G_z2 = output.mean().item()
        # G를 업데이트 합니다
        optimizerG.step()

        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(trn_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if epoch%10==0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(make_grid(fake, nrow=int(b_size**.5), padding=2, normalize=True).permute(1,2,0))
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
##
# plt.figure()
# plt.plot(G_losses)
# plt.plot(D_losses)

# import matplotlib.animation as animation
# from IPython.display import HTML
# ims = [[plt.imshow(i, animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# HTML(ani.to_jshtml())