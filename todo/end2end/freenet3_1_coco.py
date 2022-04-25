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

parser.add_argument('--max_epochs',     default=100,      type=int,      help = '')
parser.add_argument('--batch_size',     default=160,     type=int,      help = '')
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
image_list, id2ctg, ctg2name = get_coco_list(ano_fnm, limit=200, repeat=0)
ctg2id = {list(id2ctg.values())[i]:list(id2ctg.keys())[i] for i in range(len(id2ctg))}
id2name = {i:ctg2name[id2ctg[i]] for i in id2ctg.keys()} # 1~80
id2name[0]='background'

train_list_, test_list =  train_test_split(image_list,test_size=0.15164)
train_list, valid_list = train_test_split(train_list_,test_size=0.2)
##
alb = A.Compose([A.Resize(224,224),
                 A.Normalize(),
                 ToTensorV2()],
                bbox_params=A.BboxParams(format='coco',
                                         label_fields=['category_ids']),)

pretrained = models.resnet18(pretrained=True)
key = 'layer2,layer3,layer4'.split(',')
trnset = coco(train_list, model=pretrained, trans=alb, key=key)
valset = coco(valid_list, model=pretrained, trans=alb, key=key)
tstset = coco(test_list, model=pretrained, trans=alb, key=key)
trn_loader = DataLoader(trnset,batch_size=hparams.batch_size, shuffle=True,  drop_last=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(valset,batch_size=hparams.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
#zz = next(iter(DataLoader(trnset,batch_size=64, shuffle=True,  drop_last=True, num_workers=0, pin_memory=True)))
##
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
                max_epochs=hparams.max_epochs,
                gpus = 2,
                logger=tube,
                deterministic=True, accelerator='dp', accumulate_grad_batches=2)

print('Train model...')

model = freenet3_1(hparams)
trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

best_pth = checkpoint_callback.kth_best_model_path
##
best_pth = glob(r'D:\cv\free\log3\*\checkpoints/*.ckpt')[-1]
result = pd.read_csv(r'%s/../../metrics.csv'%best_pth, usecols=['epoch','trn_loss','val_loss'])
result.groupby('epoch').mean().plot(marker='*',color=['orange','b'])
#
#model = freenet(hparams)
model = freenet3_1(hparams)
model = model.load_from_checkpoint(best_pth, hparams=hparams)
print(best_pth)
##
phase = 'test'
pick = 10
p_threshold = 5
c_threshold = 0.0
#
resize_n = 672
max_n = {'train':640, 'valid':160, 'test':143}
for n in np.random.randint(0,max_n[phase],pick):
    #
    if phase=='train':
        dset = trnset[n]
        info = train_list[n]
    elif phase=='valid':
        dset = valset[n]
        info = valid_list[n]
    else:
        dset = tstset[n]
        info = test_list[n]
    for key in '2,3,4'.split(','):
        dset[key] = dset[key].unsqueeze(0)

    target_p,target_bb,target_c = dset['targets']
    # coco_decode(target_bb,[target_no])
    model.eval()
    output = model(dset)

    img_raw = cv.imread(r'%s/%012d.jpg' % (root_dir, info['image_id']))
    img = A.Resize(resize_n,resize_n)(image=img_raw)['image']/255
    #h, w, _ = img_raw.shape
    h,w = resize_n,resize_n

    grt = (target_bb*resize_n).int().numpy() ############################################################
    img = cv.rectangle(img, grt[:2],grt[:2]+grt[2:],color=[0,0,1],thickness=2)
    img = cv.putText(img, id2name[info['labels'][0]], grt[:2]+np.array([10,20]), fontFace=cv.FONT_ITALIC, fontScale=0.8, color=[0, 0, 1], thickness=2)

    out_score,out_p = torch.max(nn.Softmax(dim=-1)(output[0]),dim=-1)

        # out_score,out_c = torch.max(nn.Softmax(dim=1)(out_c_),dim=1)
        # if (out_c!=0).sum()==0:
        #     print('Fail')
        # pred_bb_ = out_bb[out_c!=0]
        # pred_score_ = out_score[out_c != 0]
        # pred_c_ = out_c[out_c != 0]
        #
        # # bb wh 오측 제거
        # bb_ind = ((pred_bb_[:, 2:] > 0).all(dim=-1)) &  ((pred_bb_[:,:2] <1).all(dim=-1))
        # pred_bb_ = pred_bb_[bb_ind]
        # pred_score_ = pred_score_[bb_ind]
        # pred_c_ = pred_c_[bb_ind]
        #
        # if pred_score_.size(0)>0:
        #     ind = torch.argsort(pred_score_,descending=True)[:p_threshold]
        #     pred_bb = pred_bb_[ind]
        #     pred_score = pred_score_[ind].numpy()
        #     pred_c = pred_c_[ind].numpy()
        #
        #     # rect pred bboxes
        #     if pred_bb.size(0) > 0:
        #         pred_bbox = pred_bb.clone()
        #         pred_bbox[:, 2:]+=pred_bbox[:, :2]
        #         pred_bbox = torch.clamp(pred_bbox,min=0.01,max=0.99)
        #         #
        #         pred_bbox[:, [0, 2]] *= w
        #         pred_bbox[:, [1, 3]] *= h
        #         pred_bbox = pred_bbox.long().numpy()
        #
        #         for bb,c,sc in zip(pred_bbox,pred_c,pred_score):
        #             lt = bb[:2]
        #             rb = bb[2:]
        #             img = cv.rectangle(img, lt, rb,color=[0,1,0],thickness=1)
        #             org = lt + np.array([-30, -10])
        #             img = cv.putText(img,'Size%02d_%s : %2.3f'%(i,id2name[c],sc),org,fontFace=cv.FONT_ITALIC, fontScale=0.5,color=[0,1,0],thickness=1)
        #
        #         tmp = []
        #         for p in pred_score:
        #             tmp.append(id2name[pred_c[pred_score == p].item()])
        #         print(id2name[info['labels'][0]], '->', i,tmp)
                #print('\t',[*map(lambda x:'%02d'%x,out_c.numpy())])

    pred_p = out_p.clone()
    pred_p[(out_p == 1) & (out_score < 0.5)] = 0


    pred_up = nn.Upsample(scale_factor=h/pred_p.size(0),mode='nearest')(pred_p.unsqueeze(0).unsqueeze(0).type(torch.float64))[0,0]
    heatmap = pred_up.unsqueeze(-1).expand_as(torch.tensor(img)).numpy()
    img_h = cv.addWeighted(heatmap,0.3,img,0.7,0.1)
    cv.imshow('%s_%s' % (id2name[info['labels'][0]], n), img_h)
    print(pred_p.sum())

