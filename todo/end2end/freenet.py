import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
from glob import glob as glob
#
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchvision import models
from torchvision import transforms
#
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
#
import cv2 as cv
from PIL import Image as pil
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
from todo.end2end.utils import *
## Image net tensor Image load
seed_everything()
# segment 개념으로 출발
# 각 class마다 따로 segment
# 동일 segment에서, 여러 객체가 있을 경우 2~N, 없을 경우 0
# 겹치는 지역은 1로 따로 표시...?

img_dir = r'D:\cv\Dataset\Imagenet\img_tensor'
label_dir = r'D:\cv\Dataset\Imagenet\label_tensor'

img_fnm = glob(f'{img_dir}/*')
label_fnm = glob(f'{label_dir}/*')

trnx,valx,trny,valy = train_test_split(img_fnm,label_fnm,test_size=0.2)

trnset = get_imagenet_tensor(trnx,trny)
valset = get_imagenet_tensor(valx,valy)

trnset[0]['labels_mat'].shape

##
max_epochs = 100
batch_size = 64
trn_loader = DataLoader(trnset,batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(valset,batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
#
sdir = r'D:\cv\free'
name     = 'log'
log_path = '%s'%(sdir)
tube     = Tube(name=name, save_dir=log_path)
checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      filename='{epoch:02d}-{val_loss:.4f}',
                                      save_top_k=1,
                                      mode='min')
early_stopping        = EarlyStopping(monitor='val_loss',
                                      patience=20,
                                      verbose=True,
                                      mode='min')
lr_monitor = LearningRateMonitor(logging_interval='epoch')


print('Call trainer...')
trainer=Trainer(callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                max_epochs=max_epochs,
                gpus = 2,
                logger=tube,
                deterministic=True, accelerator='dp', accumulate_grad_batches=2)

print('Train model...')

model = res()
trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

best_pth = checkpoint_callback.kth_best_model_path
##

best_pth = glob(r'D:\cv\free\log\*\checkpoints/*')[-1]
result = pd.read_csv(r'%s/../../metrics.csv'%best_pth, usecols=['trn_loss','val_loss','epoch'])
result.groupby('epoch').mean().plot()
#
model = res()
model = model.load_from_checkpoint(best_pth)
##
for n in np.random.randint(0,10000,5):
    dset = valset[n]
    name = valx[n][-11:-3]
    #
    z = dset['imgs'].unsqueeze(0)
    target = dset['labels_mat']
    #
    model.eval()
    output = model(z)
    output = nn.Softmax(dim=0)(output)
    score, pred = torch.max(output,dim=0)
    pred[score<.5] = 0
    #
    img_raw = cv.imread(glob('D:\cv\Dataset\Imagenet\ILSVRC2012_img_val/*%s*'%name)[0])
    img = transforms.Resize([56,56])(pil.fromarray(img_raw)).__array__()
    #
    cmap = plt.get_cmap('viridis',1002)
    fig = plt.figure(figsize=[12,8])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img,alpha=0.4)
    c1 = ax1.imshow(pred.numpy(),vmin=1,vmax=1001, cmap=cmap,alpha=0.7)
    c1.cmap.set_under('w',alpha=0.1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img,alpha=0.8)
    c2 = ax2.imshow(target,vmin=1,vmax=1001, cmap=cmap,alpha=0.7)
    #c2.cmap.set_under('w')
    cax = fig.add_axes([0.1,0.1,0.8,0.03])
    fig.colorbar(c1,cax = cax, orientation='horizontal')

    #plt.imshow(pred.cpu().numpy())