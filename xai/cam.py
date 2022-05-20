import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
from glob import glob as glob
#
import torch
from torch import nn
from torch import optim as opt
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision import models
#
import cv2 as cv
from PIL import Image as pil
from tqdm import tqdm as tqdm
from xai.utils import *
## VOC Image load
seed_everything()

fdir = r'D:\cv\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
trans = transforms.Compose([transforms.Resize([224,224]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

trnset = voc_dataset(fdir, phase='train', trans=trans, imshow=False)
valset = voc_dataset(fdir, phase='valid', trans=trans, imshow=False)

##
batch_size = 64

trn_loader = DataLoader(trnset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
## cam, gradcam
num_classes = 20
max_epochs = 1
model_cam = vgg(num_classes = num_classes)
model_cam.cuda()


optimizer = opt.AdamW(model_cam.parameters(), lr = 1e-5, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

loss_train = []
for ep in range(max_epochs):
    loss_epoch = 0
    wrap = tqdm(trn_loader)
    for batch in wrap:
        optimizer.zero_grad()
        x = batch['imgs'].cuda()
        target = batch['labels'].cuda() # test니까 하나만 학습
        pred = model_cam(x)
        loss = criterion(pred,target)
        loss.backward()
        optimizer.step()
        loss_epoch+=loss.detach().cpu()
        del x,target
        gc.collect()
        wrap.set_postfix({
            'Epoch': ep + 1,
            'Mean Loss': '{:04f}'.format(loss)})
    loss_epoch/=len(trn_loader)
    loss_train.append(loss_epoch)
    del wrap
    gc.collect()
plt.plot(loss_train)
# batch
for batch in trn_loader:
    break
## get cam
img = batch['imgs']
label = batch['labels']

model_cam.cpu()
model_cam.eval()

# using cpu
result_c = cam(model_cam, img, label)
result_c.show(1)
## get grad cam
net = models.vgg11(pretrained=True)
target_layer = net.features[-1]
model_grad = vgg_grad(net, target_layer)

result_grad = gradcam(model_grad,img,label)
result_grad.show(1)