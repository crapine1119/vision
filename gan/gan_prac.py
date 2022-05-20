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
import cv2 as cv
from glob import glob as glob
from tqdm import tqdm
from torchvision import transforms
from PIL import Image as pil
from gan.modules import *
from torchvision.utils import make_grid
##
fdir = r'D:\cv\Dataset\mpii_human_pose_v1.tar\images'
train = glob(f'{fdir}/*')[:10000]

trnset = custom(train)
b_size = 128
trn_loader = DataLoader(trnset, batch_size=b_size, drop_last=True, shuffle=True, num_workers=0, pin_memory=True)

for real_batch in trn_loader:
    break
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
lr = 2e-4
num_epochs = 100

criterion = nn.BCELoss()
fixed_noise = torch.randn(b_size, nz, 1, 1).cuda()
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
netD.train()
netG.train()
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(tqdm(trn_loader), 0):
        # 준비물
        real = data.cuda() # N,3,64,64
        label_r = torch.full((b_size,), 1., dtype=torch.float).cuda() # label.fill_(real_label)  # fake labels are real for generator cost
        label_f = torch.full((b_size,), 0., dtype=torch.float).cuda() # label.fill_(real_label)  # fake labels are real for generator cost
        # 1. 찐 이미지 확인
        netD.zero_grad()
        outD_r = netD(real).view(-1)                # 찐 판별
        loss_d_r = criterion(outD_r, label_r)
        loss_d_r.backward()
        # 2. 노이즈
        noise = torch.randn(b_size, nz, 1, 1).cuda()
        fake = netG(noise) # N,3,64,64
        outD_f = netD(fake.detach()).view(-1) # N,1,1,1
        loss_d_f = criterion(outD_f, label_f)
        loss_d_f.backward()
        loss_d = (loss_d_r+loss_d_f)/2
        optimizerD.step()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ noise로 fake 이미지 생성 : 진짜처럼
        netG.zero_grad()
        # fake 판별 결과
        outD_f = netD(fake).view(-1)
        loss_g = criterion(outD_f, label_r)  # G를 1로 학습
        loss_g.backward()
        optimizerG.step()

        if i % 10 == 0:
            print('%s : G(%2.4f), D(%2.4f)'%(epoch,loss_g.item(),loss_d.item()))

        if epoch%10==0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                img_list.append(make_grid(fake, nrow=int(b_size**.5), padding=2, normalize=True).permute(1,2,0))
        # Save Losses for plotting later
        G_losses.append(loss_g.item())
        D_losses.append(loss_d.item())
##
plt.figure()
plt.plot(G_losses)
plt.plot(D_losses)

plt.figure()
plt.imshow(img_list[-1].numpy())

noise = torch.randn(b_size, nz, 1, 1).cuda()
netG.eval()
with torch.no_grad():
    fake_imgs = netG(noise)
##
exam = make_grid(fake_imgs.detach().cpu(), nrow=int(b_size**.5), padding=2, normalize=True).permute(1,2,0)

plt.imshow(exam.numpy())
