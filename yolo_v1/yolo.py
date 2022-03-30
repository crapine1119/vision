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
from torchvision import transforms
from torchvision import models
from torch import nn
import torch.optim as opt
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader,TensorDataset,Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import cv2 as cv
import json
sys.path.append(r'C:\Users\82109\PycharmProjects\PJ\5.Job\paper_recall\yolo_v1')
import glob
from tqdm import tqdm
from torchvision.ops import nms
from modules import *
##
seed_everything(seed=42)

root_dir = r'D:\cv\Dataset/coco_2017/val2017/'
ano_fnm = r'D:\cv\Dataset/coco_2017/annotations/instances_val2017.json'

image_list, id2ctg, ctg2name = get_items(ano_fnm, limit=1500, repeat=5)

ctg2id = {list(id2ctg.values())[i]:list(id2ctg.keys())[i] for i in range(len(id2ctg))}

train_list_, test_list =  train_test_split(image_list,test_size=0.2)
train_list, valid_list = train_test_split(train_list_,test_size=0.2)
######################################
num_classes = 5 #len(ctg2id)
num_boxes = 2

trn_set = custom(train_list, num_classes, transforms=compose())
val_set = custom(valid_list, num_classes, transforms=compose())
tst_set = custom(test_list,  num_classes, transforms=compose(train=False))


trn_loader = DataLoader(trn_set, batch_size=64, shuffle=True,  drop_last=True,  num_workers=8, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, drop_last=True,  num_workers=8, pin_memory=True)
tst_loader = DataLoader(tst_set, batch_size=64, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

from tqdm import tqdm as tqdm
import gc

########################################## 일반 torch 버전
# max_epoch = 8
# optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5, weight_decay=0.0005)
# 
# loss_total = {'train':[],'valid':[]}
# 
# batch_iter = {'train':len(trn_loader),'valid':len(val_loader)}
# 
# for epoch in range(max_epoch):
#     loss_epoch = {'train':0,'valid':0}
#     wrap = {'train': tqdm(trn_loader), 'valid': tqdm(val_loader)}
#     for k in ['train','valid']:
#         if k=='train':
#             model.train()
#         else:
#             model.eval()
#         #
#         for idx,batch in enumerate(wrap[k]):
#             optimizer.zero_grad()
#             img,label = batch['img'].cuda(),batch['label'].cuda()
#             #
#             with torch.set_grad_enabled(k == 'train'):
#                 output = model(img)
#                 loss_batch = loss_f(output, label)
#                 if k=='train':
#                     loss_batch.backward()
#                     optimizer.step()
#             loss_epoch[k]+=loss_batch
#             # train이면 valid 분모는 batch_iter[valid]
#             if k=='train':
#                 trn_idx = int(idx)+1
#                 val_idx = batch_iter['valid']+1
#             else:
#                 trn_idx = batch_iter['train']+1
#                 val_idx = int(idx)+1
# 
#             wrap[k].set_postfix({
#                 'Epoch': epoch + 1,
#                 'Mean Loss': '{:04f}'.format(loss_epoch['train']/trn_idx),
#                 'Mean Val Loss': '{:04f}'.format(loss_epoch['valid']/val_idx)})
#             del img,label
#             gc.collect()
#         loss_total[k].append(loss_epoch[k]/batch_iter[k])
# 
# plt.figure()
# plt.plot(loss_total['train'])
# plt.plot(loss_total['valid'])
##
fdir = r'C:\Users\82109\PycharmProjects\PJ\5.Job\paper_recall\yolo_v1'
name     = 'log'
log_path = '%s'%(fdir)
tube     = Tube(name=name, save_dir=log_path)
checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      filename='{epoch:02d}-{val_loss:.4f}',
                                      save_top_k=1,
                                      mode='min')
early_stopping        = EarlyStopping(monitor='val_loss',
                                      patience=10,
                                      verbose=True,
                                      mode='min')
lr_monitor = LearningRateMonitor(logging_interval='epoch')

print('Call trainer...')
trainer=Trainer(callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                max_epochs=100,
                gpus = 2,
                logger=tube,
                deterministic=True, accelerator='dp', accumulate_grad_batches=2)

print('Train model...')

model = yolo_v1()

trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

best_pth = checkpoint_callback.kth_best_model_path
##


best_pth = glob.glob(r'%s\checkpoints/*'%(glob.glob(r'C:\Users\82109\PycharmProjects\PJ\5.Job\paper_recall\yolo_v1\log/*')[-1]))[0]
#
fortest = yolo_v1()
fortest = fortest.load_from_checkpoint(best_pth)
fortest = nn.DataParallel(fortest).cuda()

#
def decode(output, target, conf_threshold = 0.2, prob_threshold=0.1):
    """
    :param output: : dimension {7,7,5*bboxes+classes}
    :param target:
    :return:
    """
    point_ind = target.sum(dim=-1)!=0
    out_box, out_label, out_conf, out_score = [],[],[],[]
    # 모델 결과 분류
    class_score,class_label = torch.max(output[:,:,10:],-1)
    conf = output[:,:,[4,9]].clone()
    prob = conf*class_score.unsqueeze(-1).repeat(1,1,2)
    boxes = output[:,:,torch.LongTensor([0,1,2,3,5,6,7,8])] # 7*7*2
    # 셀에서 좌측 상단 좌표 구하기
    cell_x,cell_y = np.meshgrid(np.arange(7),np.arange(7)) # 7*7*2(x,y)
    lt_norm_ = np.dstack([cell_x,cell_y])* 1/7 # 일단 좌측 상단 기준 : 7*7*2(x,y)
    lt_norm = np.dstack([lt_norm_,lt_norm_]) # 7*7*4 (x,y,x,y)
    #
    boxes_target = target[point_ind].squeeze()[:4]
    label_target = torch.argsort(target[point_ind].squeeze()[10:])[-1].item()+1

    # xy(셀 기준) -> xy(이미지 기준)
    xy_norm = boxes[:,:,[0,1,4,5]]* 1/7 + lt_norm
    wh_norm = boxes[:,:,[2,3,6,7]].clone()

    xy_norm_target =  boxes_target[:2]*1/7 + lt_norm[point_ind].squeeze()[:2]
    wh_norm_target = boxes_target[2:].clone()

    # b1, b2에서의 좌상/우하 좌표
    lt_b1b2 = xy_norm - wh_norm/2 # b1 lt, b2 lt
    rb_b1b2 = xy_norm + wh_norm/2 # b1 rb, b2 rb
    lt_target = xy_norm_target - wh_norm_target / 2  # b1 lt, b2 lt
    rb_target = xy_norm_target + wh_norm_target / 2  # b1 rb, b2 rb

    # bbox1, bbox2
    bbox1 = torch.cat([lt_b1b2[:,:,:2],rb_b1b2[:,:,:2]],dim=-1)
    bbox2 = torch.cat([lt_b1b2[:,:,2:],rb_b1b2[:,:,2:]],dim=-1)
    bbox_target = torch.cat([lt_target,rb_target])
    """
    class_label.size() # 7*7 각 셀에서의 label
    conf.size() # bbox1,2에서의 conf(iou)
    prob.size() # bbox1,2에서의 prob(iou*class_score)
    bbox1.size() # bbox1 (이미지에서의 비율)
    bbox2.size() # bbox2 (")
    """
    # 확률 0.05 이상
    bbox1_ind = (prob[:,:,0]>=prob_threshold) & (class_score>conf_threshold)
    bbox2_ind = (prob[:,:,1]>=prob_threshold) & (class_score>conf_threshold)

    if bbox1_ind.sum()>0:
        out_box.append(bbox1[bbox1_ind])
        out_label.append(class_label[bbox1_ind])
        out_conf.append(conf[:,:,0][bbox1_ind])
        out_score.append(class_score[bbox1_ind])
    if bbox2_ind.sum()>0:
        out_box.append(bbox2[bbox2_ind])
        out_label.append(class_label[bbox2_ind])
        out_conf.append(conf[:,:,1][bbox2_ind])
        out_score.append(class_score[bbox2_ind])
    if len(out_box)>0:
        out_box   = torch.cat(out_box,dim=0)
        out_label = torch.cat(out_label, dim=0)
        out_conf  = torch.cat(out_conf, dim=0)
        out_score = torch.cat(out_score, dim=0)
        #
        out_box = out_box.type(torch.FloatTensor)
        out_label = out_label.type(torch.LongTensor)
        out_conf = out_conf.type(torch.FloatTensor)
        out_score = out_score.type(torch.FloatTensor)
    else:
        print(prob.max())
    return out_box,out_label,out_conf,out_score, bbox_target, label_target
#
def nms_(out_box,out_score):
    if len(out_box)>0:
        x1 = out_box[:, 0]
        y1 = out_box[:, 1]
        x2 = out_box[:, 2]
        y2 = out_box[:, 3]
        areas = (x2-x1) * (y2-y1)
        _,ids_sorted = out_score.sort(0,descending=True)
        ids = []
        # while ids_sorted.numel()>1:
        #     i = ids_sorted[0].item()
        #     ids_sorted = ids_sorted[1:]
            #x1[i]
            #torch.clamp(x1[i],min=)

#
def visualize(image, output, target, conf_threshold, prob_threshold, nms_threshold=0.1, cv_name='1'):
    image = image.permute(1, 2, 0).numpy()
    image = cv.cvtColor(image,cv.COLOR_RGB2BGR)

    out_box, out_label, out_conf, out_score, bbox_target,bbox_label = decode(output,target,conf_threshold=conf_threshold, prob_threshold=prob_threshold)
    # Truth
    rx1, ry1, rx2, ry2 = (bbox_target * 448).int().tolist()
    image = cv.rectangle(image, (rx1, ry1), (rx2, ry2), [0, 0, 1], 2)
    image = cv.putText(image, '%s' % (ctg2name[bbox_label]),
                       org=(rx1 + 10, ry1 + 20),
                       fontFace=cv.FONT_ITALIC,
                       fontScale=0.4,
                       color=[0, 0, 1],
                       thickness=1)
    if len(out_box)>0:
        nms_ind = nms(out_box,out_score,nms_threshold)
        if len(nms_ind)>0:
            n = len(nms_ind)
            for i in range(n):
                xy = out_box[nms_ind][i].clone()
                xy[:2] = torch.clamp(xy[:2],min=0.01)
                xy[2:] = torch.clamp(xy[2:], max=0.99)
                iou = Loss.get_iou(Loss, xy.unsqueeze(0), bbox_target.unsqueeze(0)).item()

                x1,y1,x2,y2 = (xy*448).int().tolist()
                fig_name = ctg2name[out_label[nms_ind][i].item()+1]
                score = out_score[nms_ind][i]

                image = cv.rectangle(image,(x1,y1),(x2,y2),[0,1,0],2)
                image = cv.putText(image,'%s : %2.2f (%2.2f)'%(fig_name,score,iou),
                                   org=(x1-20,y1+20),
                                   fontFace=cv.FONT_ITALIC,
                                   fontScale=0.4,
                                   color=[0,1,0],
                                   thickness=1)

    cv.imshow(cv_name, image)
##
ref_ths = 0.02
for n in np.random.randint(0,1430,10):
    # img = tst_set[n]['img'].unsqueeze(0).cuda()
    # target = tst_set[n]['label']
    img = trn_set[n]['img'].unsqueeze(0).cuda()
    target = trn_set[n]['label']

    fortest.eval()
    output = fortest(img)
    output = output.squeeze().detach().cpu()
    score,label = torch.max(output[:, :, 10:], dim=-1)
    higher = [4, 9][np.argmax([output[:, :, 4].max(), output[:, :, 9].max()])]

    prob_max = (score.unsqueeze(-1)*output[:,:,[4,9]]).max()
    output[:,:,4]
    # if prob_max<ref_ths:
    #     print('pass')
    # else:
    #     print(label)
    image = img.detach().cpu().squeeze()
    visualize(image, output, target, conf_threshold=0.5, prob_threshold=ref_ths, nms_threshold=0.0, cv_name=str(n))
    #visualize(image, target, target, conf_threshold=0.2, prob_threshold=0.001, nms_threshold=0.5, cv_name=str(n))

##




##
import pandas as pd
from matplotlib import pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


pd.read_csv(r'C:\Users\82109\PycharmProjects\PJ\5.Job\paper_recall\yolo_v1\log\version_5/metrics.csv').groupby('epoch').mean()[['trn_loss','val_loss']].plot(ax=ax1)
