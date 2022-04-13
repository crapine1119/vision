import numpy as np
import pandas as pd
import os
import gc
import argparse
import matplotlib.pyplot as plt
from glob import glob as glob
#
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
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
##
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--out_size',       default=128,     type=int,      help = '')

parser.add_argument('--max_epochs',     default=50,      type=int,      help = '')
parser.add_argument('--batch_size',     default=256,     type=int,      help = '')
#
parser.add_argument('--lr',             default=0.01,   type=float,    help = '')
parser.add_argument('--wd',             default=0.005,   type=float,    help = '')
parser.add_argument('--lr_patience',    default=5,       type=int,    help = '')
parser.add_argument('--es_patience',    default=10,      type=int,    help = '')

parser.add_argument('--random_seed',    default=42,      type=int,      help = '')
# hparams = parser.parse_args()
hparams = parser.parse_args(args=[]) # 테스트용


## Image net tensor Image load
seed_everything(seed=hparams.random_seed)

root_dir = r'D:\cv\Dataset/coco_2017/val2017/'
ano_fnm = r'D:\cv\Dataset/coco_2017/annotations/instances_val2017.json'

# image_list, id2ctg, ctg2name = get_items(ano_fnm, limit=1500, repeat=5)
image_list, id2ctg, ctg2name = get_coco_list(ano_fnm, limit=100, repeat=0)
ctg2id = {list(id2ctg.values())[i]:list(id2ctg.keys())[i] for i in range(len(id2ctg))}
id2name = {i:ctg2name[id2ctg[i]] for i in id2ctg.keys()} # 1~80

train_list_, test_list =  train_test_split(image_list,test_size=0.2)
train_list, valid_list = train_test_split(train_list_,test_size=0.2)
##
alb = A.Compose([A.Resize(224,224),
                 A.Normalize(),
                 ToTensorV2()],
                bbox_params=A.BboxParams(format='coco',
                                         label_fields=['category_ids']),)

pretrained = models.resnet18(pretrained=True)
trnset = coco(train_list, model=pretrained, trans=alb)
valset = coco(valid_list, model=pretrained, trans=alb)

##
trn_loader = DataLoader(trnset,batch_size=hparams.batch_size, shuffle=True,  drop_last=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(valset,batch_size=hparams.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)

#
sdir = r'D:\cv\free'
name     = 'log3'
log_path = '%s'%(sdir)
tube     = Tube(name=name, save_dir=log_path)
checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      filename='{epoch:02d}-{val_loss:.4f}',
                                      save_top_k=1,
                                      mode='min')
early_stopping        = EarlyStopping(monitor='val_loss',
                                      patience=hparams.es_patience,
                                      verbose=True,
                                      mode='min')
lr_monitor = LearningRateMonitor(logging_interval='epoch')


print('Call trainer...')
trainer=Trainer(callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                max_epochs=1,
                gpus = 2,
                logger=tube,
                deterministic=True, accelerator='dp', accumulate_grad_batches=2)

print('Train model...')

model = freenet3(hparams)
trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

best_pth = checkpoint_callback.kth_best_model_path
##
best_pth = glob(r'D:\cv\free\log2\*\checkpoints/*')[-1]
result = pd.read_csv(r'%s/../../metrics.csv'%best_pth, usecols=['epoch','trn_loss','val_loss'])
result.groupby('epoch').mean().plot(marker='*')
#
#model = freenet(hparams)
model = freenet3(hparams)
model = model.load_from_checkpoint(best_pth, hparams=hparams)
print(best_pth)
##
phase = 'train'
pick = 15
p_threshold = 5
c_threshold = 0.0
#
resize_n = 672
max_n = {'train':4800, 'valid':1200}
for n in np.random.randint(0,max_n[phase],pick):
    #
    if phase=='train':
        dset = trnset[n]
        info = train_list[n]
    else:
        dset = valset[n]
        info = valid_list[n]
    for key in '1,2,3,4'.split(','):
        dset[key] = dset[key].unsqueeze(0)
    target_p,target_bb,target_c = dset['targets']

    model.eval()
    output = model(dset)

    # 물체 위치, 물체일 확률
    out_p,out_bb,out_c,out_nms = output

    out_c_score,_  = torch.max(nn.Softmax(dim=-1)(out_c[0]),dim=-1)
    out_p_score, _ = torch.max(nn.Softmax(dim=-1)(out_p[0]), dim=-1)

    _, out_p = torch.max(out_p[0], dim=-1)
    _, out_c = torch.max(out_c[0], dim=-1)
    _, out_nms = torch.max(out_nms[0], dim=-1)
    # 물체가 있는 곳, 확률이 가장 높은 n개
    best_num = torch.argsort((out_p_score * out_c_score * out_p).flatten(),descending=True)[:p_threshold] # ROI score(Positive) * Class score(Positive)
    #best_num = torch.argsort((out_c_score * out_p).flatten(),descending=True)[:p_threshold] # Class score(Positive)
    best_ind = torch.zeros_like(out_p_score).flatten().bool()
    best_ind[best_num] = True

    #act_p = out_p.bool() & (out_c!=0) & out_nms.bool() & best_ind.view(56,56) # (positive, not bg class, nms, best n)
    act_p = out_p.bool() & (out_c!=0) & best_ind.view(56,56) # (positive, not bg class, nms, best n)
    print('No.%s Positive C : '%n, act_p.sum())
    print('roi:',out_p.sum().item(),'nms:',out_nms.sum().item())
    # 인덱싱
    pred_c_ = out_c[act_p]
    pred_score_ = out_c_score[act_p]
    xx, yy = np.meshgrid(np.arange(0, 56, 1), np.arange(0, 56, 1))
    cxy = (np.dstack([xx, yy])+.5) / 56
    pred_cxy_ = torch.FloatTensor(cxy)
    pred_cxy_ = pred_cxy_[act_p]

    pred_bb = out_bb[0, act_p]
    pred_bb = torch.clamp(pred_bb,min=0,max=1)
    #
    bb_true = (pred_bb[:,2:]==0).sum(dim=1)==0
    pred_cxy = pred_cxy_[bb_true]
    pred_wh = pred_bb[bb_true,2:].detach()
    pred_c = pred_c_[bb_true].detach().numpy()
    pred_score = pred_score_[bb_true].detach().numpy()


    img_raw = cv.imread(r'%s/%012d.jpg' % (root_dir, info['image_id']))
    img = A.Resize(resize_n,resize_n)(image=img_raw)['image']/255
    #h, w, _ = img_raw.shape
    h,w = resize_n,resize_n

    grt = (target_bb*resize_n).int().numpy()
    img = cv.rectangle(img, grt[:2],grt[:2]+grt[2:],color=[0,0,1],thickness=2)
    img = cv.putText(img, id2name[info['labels'][0]], grt[:2]+np.array([10,20]), fontFace=cv.FONT_ITALIC, fontScale=0.8, color=[0, 0, 1], thickness=2)

    # rect pred bboxes
    if len(pred_wh) > 0:
        pred_cxy *= w
        pred_cxy = pred_cxy.long().numpy()

        pred_wh[:, 0] *= w
        pred_wh[:, 1] *= h
        pred_wh = pred_wh.long().numpy()

        for cxy,wh,c,sc in zip(pred_cxy,pred_wh,pred_c,pred_score):
            lt = cxy - wh // 2
            rb = cxy + wh // 2
            lt[lt <= 0] = 1
            rb[lt >= h] = h-1

            img = cv.rectangle(img, lt, rb,color=[0,1,0],thickness=1)
            org = lt + np.array([10, 20])
            img = cv.putText(img,'%s : %2.3f'%(id2name[c],sc),org,fontFace=cv.FONT_ITALIC, fontScale=0.5,color=[0,1,0],thickness=2)

    act_p_ = .7*torch.cat([torch.zeros_like(act_p.unsqueeze(0)).repeat(2, 1, 1), act_p.unsqueeze(0)], dim=0)
    out_p_ = .3*out_p.unsqueeze(0).repeat(3,1,1)

    heatmap = act_p_+out_p_
    heatmap = nn.Upsample(scale_factor=resize_n/heatmap.size(-1),mode='nearest')(heatmap.unsqueeze(0))[0].permute(1,2,0).numpy().astype(np.float64)
    img = cv.addWeighted(heatmap,0.7,img,0.7,0)
    cv.imshow('%s_%s' % (id2name[info['labels'][0]], n), img)
    print(id2name[info['labels'][0]],'->', id2name[out_c[out_c_score == out_c_score.max()].item()+1])