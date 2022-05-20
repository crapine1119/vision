import pandas as pd
import json
import numpy as np
from glob import glob as glob
import cv2 as cv
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models
from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import random
import os
from torchvision import transforms
from PIL import Image
import dlib
import albumentations as A
from torchvision.ops import nms
from pytorch_lightning import LightningModule
from torch import optim as opt
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
##
def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print('Seed : %s'%seed)

def get_iou(bb1, bb2):
    """
    :param bb1: must tensor {N,xywh}
    :param bb2: must tensor {N,xywh}
    :return:
    """
    # assert bb1[:,0] < bb1[:,2]
    # assert bb1[:,1] < bb1[:,3]
    # assert bb2[:,0] < bb2[:,2]
    # assert bb2[:,1] < bb2[:,3]
    #
    x_left = torch.max(bb1[:,0], bb2[:,0])
    y_top = torch.max(bb1[:,1], bb2[:,1])

    x_right = torch.min(bb1[:,0]+bb1[:,2], bb2[:,0]+bb2[:,2])
    y_bottom = torch.min(bb1[:,1]+bb1[:,3], bb1[:,1]+bb2[:,3])

    intersection_x = x_right - x_left
    intersection_y = y_bottom - y_top
    intersection_x[intersection_x < 0] = 0
    intersection_y[intersection_y < 0] = 0
    #
    intersection_area = intersection_x * intersection_y
    bb1_area = abs(bb1[:,2] * bb1[:,3])
    bb2_area = abs(bb2[:,2] * bb2[:,3])
    iou = intersection_area / (bb1_area + bb2_area - intersection_area + 1e-7)
    return iou

def get_feature(model,key,x):
    features = {}
    for i in [*model.named_children()][:-2]:
        x = i[1](x)
        if i[0] in key:
            features[i[0][-1]] = x.squeeze()
    return features

class custom(Dataset):
    def __init__(self,trnx,trny,pretrained,key,trans,size=500,train=True):
        self.samples = trnx
        self.ano = trny
        self.size = size
        self.train = train
        self.pretrained = pretrained
        self.key=key
        self.trans = trans

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = cv.imread(self.samples[i])
        raw_h,raw_w = img.shape[0], img.shape[1]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.trans:
            mat = self.trans(image=img)
            img = mat['image']
        else:
            img = cv.resize(img,(self.size,self.size))

        if self.pretrained:
            features = get_feature(self.pretrained,self.key,img.unsqueeze(0))

        if self.train:
            points = pd.read_csv(self.ano[i], header=1, delim_whitespace=True).iloc[1:-1, :].astype(float)
            points['num'] =points.index
            #points = points[[38,44,30,60,54],:]
            points.iloc[:, 0] /= raw_w
            points.iloc[:, 1] /= raw_h

            index = torch.zeros((3,self.size,self.size))
            for p in points.values:
                x,y = p[:2]
                X,Y = (p[:2]*self.size).astype(int)
                X,Y = min(X,self.size-1),min(Y,self.size-1)
                c = p[-1]
                index[0, Y, X] = c
                index[1:, Y, X] = torch.FloatTensor([x,y])


            # x1,x2 = points[...,0].min(),points[...,0].max()
            # y1,y2 = points[...,1].min(),points[...,1].max()

            #points = points.astype('int')
            #features['img'] = img
            features['ano'] = index

            return features
        else:
            return {'img':img}

## modules
class squeeze(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.squeeze()

class flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.contiguous().view(x.size(0),-1)

class elwise_block(nn.Module):
    def __init__(self, in_c, out_c=128):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.add = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(out_c),
                                 nn.LeakyReLU())

    def forward(self, x):
        x = self.add(x)
        return x

class conv_3x3(nn.Module):
    def __init__(self, in_c=256, out_c=128, n_class=80):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, n_class, kernel_size=1, bias=False))

    def forward(self, x):
        x = self.classifier(x)
        return x

def get_callback(sdir,name='log',es_patience=15):
    log_path = '%s'%(sdir)
    tube     = Tube(name=name, save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='{epoch:02d}-{val_loss:.4f}',
                                          save_top_k=1,
                                          mode='min')
    early_stopping        = EarlyStopping(monitor='val_loss',
                                          patience=es_patience,
                                          verbose=True,
                                          mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    print('get (1)ckp, (2)es, (3)lr_monitor callbacks with (4)tube')
    return {'callbacks':[checkpoint_callback, early_stopping, lr_monitor],
            'tube':tube}
##
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    #
    def get_loss(self, modely, targety, func='smooth'):
        if func=='mse':
            f = nn.MSELoss()
        elif func=='entropy':
            f = nn.CrossEntropyLoss()
        elif func == 'smooth':
            f = nn.SmoothL1Loss()
        return f(modely,targety)

    def forward(self, modely, targety):
        """
        :param modely:  n,71(68+1,2),224,224
        :param targety: n,3(cls,x,y),224,224
        :return:
        """
        ind = targety[:,0]!=0 # n,224,224


        pred_c = modely[ind][:,:69]
        pred_bg = modely[~ind][:,:69]

        pred_xy = modely[ind][:,69:]

        target_c = targety[:, 0][ind].long()
        target_xy = targety[:,1:].permute(0,2,3,1)[ind]

        loss_c = self.get_loss(pred_c,target_c,func='entropy')
        loss_bg = self.get_loss(pred_bg,
                                torch.zeros(len(pred_bg),dtype=torch.long).cuda(),
                                func='entropy')
        loss_xy = self.get_loss(pred_xy,target_xy,func='mse')

        loss = loss_c + .5*loss_bg + .5*loss_xy
        return loss

class pwc(LightningModule):
    def __init__(self):
        super().__init__()
        out_c = 128
        self.block4 = elwise_block(512, out_c)
        self.block3 = elwise_block(256, out_c)
        self.block2 = elwise_block(128, out_c)
        self.upsamp2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.upsamp8 = nn.Upsample(scale_factor=8, mode='nearest')

        #
        self.classifier = conv_3x3(in_c=out_c, n_class=68+1+2)
        #
        self.init_weights()

    def forward(self,x):
        elwise = self.block4(x['4']) # 2048, 7,7 -> 128, 7,7
        elwise = self.upsamp2(elwise) # 128, 14,14
        elwise = self.block3(x['3'])+elwise
        elwise = self.upsamp2(elwise)  # 128, 28,28
        elwise = self.block2(x['2']) + elwise
        elwise = self.upsamp8(elwise)  # 128, 224,224
        #
        out = self.classifier(elwise)
        return out.permute(0,2,3,1)

    def loss_f(self, modely, targety):
        f = Loss()
        return f(modely, targety)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optimizer = opt.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=5,
                                                   min_lr=1e-7)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        target = x['ano']
        pred = self(x)
        loss = self.loss_f(pred, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('trn_loss', loss, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([op['val_loss'] for op in outputs]).mean()
        print("\n* EPOCH %s | loss :{%4.4f}" % (self.current_epoch, avg_loss))
        return {'loss': avg_loss}

