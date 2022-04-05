import pandas as pd
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2 as cv
from torch.utils.data import Dataset,DataLoader
#from selectivesearch import selective_search as ss
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
from mtcnn import MTCNN
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
seed_everything(seed=42)

## dset

ddir = r'D:\cv\Dataset\mmlab'

_,x1,x2,y1,y2 = pd.read_csv(f'{ddir}/lfpw_test_249_bbox.txt',delim_whitespace=True,header=None).loc[0].values

img = cv.imread(f'{ddir}/lfpw_testImage/001.jpg')
img = cv.rectangle(img,(x1,y1),(x2,y2),[0,0,255],2)
cv.imshow('test',img)

##
fdir = r'D:\cv\Dataset\300w\300W\01_Indoor'
name = 90
img = cv.imread('%s/indoor_%03d.png'%(fdir,name))
points = pd.read_csv('%s/indoor_%03d.pts'%(fdir,name),header=1,delim_whitespace=True).iloc[1:-1,:].astype(float).values
points = points.astype('int')

for p in points:
    img = cv.circle(img,(p[0],p[1]),2,[0,0,255])
cv.imshow(str(name),img)

## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
fnms = glob.glob('%s/*png'%fdir)[:-1]
fnms_ano = glob.glob('%s/*pts'%fdir)

trnx,tstx,trny,tsty = train_test_split(fnms,fnms_ano,test_size=0.2)

class custom(Dataset):
    def __init__(self,trnx,trny,size=500,train=True):
        self.samples = trnx
        self.ano = trny
        self.size = size
        self.train = train
    def __len__(self):
        return len(self.ano)

    def __getitem__(self, i):
        img = cv.imread(self.samples[i])
        raw_h,raw_w = img.shape[0], img.shape[1]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img,(self.size,self.size))

        points = pd.read_csv(self.ano[i], header=1, delim_whitespace=True).iloc[1:-1, :].astype(float).values
        #points = points[[38,44,30,60,54],:]
        points[:, 0] *= self.size/raw_w
        points[:, 1] *= self.size / raw_h,
        points = points.astype('int')

        if self.train:
            img = torch.tensor(img).permute(-1,0,1)
            points = torch.tensor(points)
            return {'img': img,
                    'ano': points}
        else:
            return {'img':img,
                    'ano':points}




##

"""
0 38
1 44
2 30 
3 60
4 54
"""


from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=True)

bboxes, conf, landmarks = mtcnn.detect(img,landmarks=True)

img = custom(trnx,trny,1000,train=False)[100]['img']
ano = custom(trnx,trny,1000,train=False)[100]['ano']

for e,i in enumerate(ano):
    img = cv.circle(img,(i[0],i[1]),2,[0,0,255])
    img = cv.putText(img, str(e), (int(i[0]), int(i[1])),0,0.3, [0, 0, 255])
cv.imshow('1', img)

x1,y1,x2,y2 = map(int,bboxes[0])
img = cv.rectangle(img,(x1,y1),(x2,y2),[0,255,0],2)
cv.imshow('1', img)


for e,i in enumerate(landmarks[0]):
    #img = cv.circle(img,(int(i[0]),int(i[1])),1,[0,255,0])
    img = cv.putText(img,str(e),(int(i[0]),int(i[1])),0,0.5,[0,255,0])


cv.imshow('1', img)
## fine tuning

model = MTCNN(keep_all=True)
model.onet.dense6_3 = nn.Linear(256,68)

trn_set = custom(trnx,trny,1000,train=True)
tst_set = custom(tstx,tsty,1000,train=True)

trn_loader = DataLoader(trn_set,batch_size=32,drop_last=True)
tst_loader = DataLoader(trn_set,batch_size=32,drop_last=True)

model = model.cuda() # 반드시 가장 나중에

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)


for epoch in range(10):
    wrap = {'train': tqdm(trn_loader),
            'valid': tqdm(tst_loader)}
    loss = {'train': 0.0, 'valid': 0.0}
    for phase in ['train','valid']:
        if phase=='train':
            model.train()
        else:
            model.eval()
        for idx,i in enumerate(wrap[phase]):
            optimizer.zero_grad()
            batch = i['img'].cuda()
            ano = i['ano'].cuda()  # 68
            with torch.set_grad_enabled(phase == 'train'):
                aaa = model(batch)
                #trash, trash_, landmarks = model(batch)
                batch_loss = criterion(points.flatten(), landmarks.flatten())  # n*10
                if phase=='train':
                    batch_loss.backward()
                    optimizer.step()
            loss[phase] += batch_loss * ano.size(0)
            wrap[phase].set_postfix({
                'Epoch': epoch + 1,
                'Mean Loss': '{:06f}'.format(loss['train'] / (idx + 1))})




            with torch.set_grad_enabled(phase == 'train'):
                output = model(input)
                batch_loss = criterion(output, label)  # 16개 평균 loss