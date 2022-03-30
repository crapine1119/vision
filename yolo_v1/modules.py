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
from PIL import Image as pil
import albumentations as A
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

def get_items(ano_fnm, limit=1000, repeat=5):
    with open(ano_fnm, 'r') as f:
        temp = json.loads(''.join(f.readlines()))
    f.close()
    image_list = []
    ctg_df = pd.DataFrame(temp['categories']).reset_index()
    ctg_df['index'] = ctg_df['index'] + np.ones(len(ctg_df), dtype=np.int64)
    id2ctg = dict(ctg_df.set_index('index')['id'])
    ctg2id = dict(ctg_df.set_index('id')['index'])
    ctg2name = dict(ctg_df.set_index('id')['name'])
    count = [0]*6
    label_list = []
    for a in temp['annotations']:
        image_id = a['image_id']
        bbox = np.stack(a['bbox'])
        #labels = np.array([ctg2id[l] for l in a['category_id']])
        labels = np.array([ctg2id[a['category_id']]])
        if (labels<=5) and (count[labels.item()]<limit):
            image_list.append({'image_id':image_id, 'bbox':bbox, 'labels':labels})
            count[labels.item()]+=1
            label_list.append(labels.item())
    image_list = np.asarray(image_list)
    label_list = np.array(label_list)
    low_class = np.arange(1,5+1)[np.array(count[1:])<limit]
    add = np.array([])
    for i in low_class:
        add = np.hstack([add, np.repeat(image_list[label_list == i], repeat - 1)])
    return np.hstack([image_list,add]), id2ctg, ctg2name

def get_iou(bb1, bb2):
    """
    :param bb1: must tensor {N,4}, 4 : 좌상/우하
    :param bb2: must tensor
    :return:
    """
    # assert bb1[:,0] < bb1[:,2]
    # assert bb1[:,1] < bb1[:,3]
    # assert bb2[:,0] < bb2[:,2]
    # assert bb2[:,1] < bb2[:,3]
    #
    x_left = torch.max(bb1[:,0], bb2[:,0])
    y_top = torch.max(bb1[:,1], bb2[:,1])
    x_right = torch.min(bb1[:,2], bb2[:,2])
    y_bottom = torch.min(bb1[:,3], bb2[:,3])

    intersection_x = x_right - x_left
    intersection_y = y_bottom - y_top
    intersection_x[intersection_x < 0] = 0
    intersection_y[intersection_y < 0] = 0
    #
    intersection_area = intersection_x * intersection_y
    bb1_area = abs((bb1[:,2] - bb1[:,0]) * (bb1[:,3] - bb1[:,1]))
    bb2_area = abs((bb2[:,2] - bb2[:,0]) * (bb2[:,3] - bb2[:,1]))
    iou = intersection_area / (bb1_area + bb2_area - intersection_area + 1e-7)
    return iou

def compose(train=True):
    if train:
        tool = transforms.Compose([transforms.Resize((448,448)),
                                   #transforms.RandomHorizontalFlip(p=0.5),
                                   #transforms.RandomVerticalFlip(p=0.5),
                                   transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                   transforms.RandomGrayscale(p=0.5),
                                   transforms.ToTensor()])
    else:
        tool = transforms.Compose([transforms.Resize((448, 448)),
                                   transforms.ToTensor()])
            # A.Compose([A.Resize((448,448)),
            #            A.HorizontalFlip(0.5),
            #            ])

    return tool

class custom(Dataset):
    def __init__(self,image_list, num_classes, transforms=None, train=True, root_dir=r'D:\cv\Dataset/coco_2017/val2017/'):
        self.image_list = image_list
        self.num_classes = num_classes
        self.transforms = transforms
        self.train = train
        self.root_dir = root_dir
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        info = self.image_list[item]
        id = info['image_id']
        bb = info['bbox'].astype(float).copy()
        label = info['labels'].copy()


        img = cv.imread(r'%s%012d.jpg'%(self.root_dir,id),cv.IMREAD_COLOR)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        h,w = img.shape[:2]
        if self.transforms:
            img = self.transforms(pil.fromarray(img))
        # 좌측상단, 크기
        bb[[0, 2]] /= float(w)
        bb[[1, 3]] /= float(h)
        if self.train:
            target = encode(bb,label,self.num_classes)
            target = torch.tensor(target,dtype=torch.float)
            return {'img':img,
                    'label':target}
        else:
            return {'img':img}

def encode(bb,label,num_classes):
    """
    :param bb: 이미지 사이즈에 대한 상대 좌표, 크기 {lt, wh}
    :param label:
    :param num_classes:
    :return:
    """
    S = 7
    B = 2
    N = 5*B + num_classes
    cell_size = 1/S
    bb[:2]+=bb[2:]/2
    target = np.zeros((S,S,N))

    cxy, wh = bb[:2],bb[2:] # 전체 이미지에 대한 상대값

    ij = (cxy * S).astype(int) # For searching the index of the lt, mult 7
    i, j = ij
    top_left = ij * cell_size # lt index -> lt relative coords
    xy_cell = (cxy - top_left) / cell_size # normalized by cell size
    for b in range(B):
        target[j,i,b*5:(b+1)*5] = np.hstack([xy_cell,wh,1])
    target[j,i,B*5+label-1] = 1 # label이 1부터 시작하므로 1을 빼줌
    return target

#def decode():

class squeeze(nn.Module):
    def __init(self):
        super().__init__()
    def forward(self,x):
        return x.squeeze()

class flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.view(x.size(0),-1)

class unflatten(nn.Module):
    def __init__(self,num_bboxes,num_classes):
        super().__init__()
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
    def forward(self,x):
        return x.view(x.size(0),5*self.num_bboxes+self.num_classes,7,7)

class Loss(nn.Module):
    def __init__(self, num_bboxes, num_classes):
        super().__init__()
        self.S = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        cell_x, cell_y = np.meshgrid(np.arange(7), np.arange(7))  # 7*7*2(x,y)
        lt_norm = np.dstack([cell_x, cell_y]) * (1 / self.S)  # 일단 좌측 상단 기준 : 7*7*2(x,y,x,y)
        self.lt_norm = torch.tensor(lt_norm, dtype=torch.float64).unsqueeze(0).cuda() # 1,7,7,2 (n개로 인덱싱 하기 위해 복사)

    def get_iou(self, bb1, bb2):
        """
        :param bb1: must tensor {N,4}, 4 : 좌상/우하
        :param bb2: must tensor "
        :return:
        """
        x_left = torch.max(bb1[:, 0], bb2[:, 0])
        y_top = torch.max(bb1[:, 1], bb2[:, 1])
        x_right = torch.min(bb1[:, 2], bb2[:, 2])
        y_bottom = torch.min(bb1[:, 3], bb2[:, 3])

        intersection_x = x_right - x_left
        intersection_y = y_bottom - y_top
        intersection_x[intersection_x < 0] = 0.0
        intersection_y[intersection_y < 0] = 0.0
        #
        intersection_area = intersection_x * intersection_y
        bb1_area = abs((bb1[:, 2] - bb1[:, 0]) * (bb1[:, 3] - bb1[:, 1]))
        bb2_area = abs((bb2[:, 2] - bb2[:, 0]) * (bb2[:, 3] - bb2[:, 1]))
        iou = intersection_area / (bb1_area + bb2_area - intersection_area + 1e-6)
        return iou
    #
    def mse(self, modely, targety):
        f = nn.MSELoss(reduction='sum')
        return f(modely,targety)

    def forward(self, modely, targety):
        """
        :param modely: n, 7,7, 5*b+c [{x,y,w,h,conf}*2 + c]
        :param targety:
        :return:
        """
        num_bboxes, num_classes = self.num_bboxes, self.num_classes

        # 타겟에서 물체 있/없는 마스킹
        coord_mask = targety[:, :, :, 4] > 0   # conf>0 : 실제 물체 있는 셀 (N, 7,7) // 4말고 9로 해도 똑같음
        noobj_mask = targety[:, :, :, 4] == 0  # 없는 곳
        # ** 인덱싱은 무슨 짓을 해도 텐서를 copy하지 않는다, 슬라이싱은 무조건 copy

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ loss tensor1. background {n*48,2}: conf가 0를 만들 수 있는지만 비교하면 됨
        noobj_conf_mask = [5 * b - 1 for b in range(1, num_bboxes + 1)] # {...,[4,9]} conf (iou)
        mat_noobj_pred = modely[noobj_mask][:, noobj_conf_mask]     # {n*48,2}
        mat_noobj_target = targety[noobj_mask][:,noobj_conf_mask]

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ loss tensor2. coord 중 높은 iou를 갖는 bbox indexing {n,10}
        iou_pred =   modely[coord_mask][:, :10]   # {n*1, xywhp} -> {n*1,p}
        iou_target =   targety[coord_mask][:, :5]

        # center x,y (relative to cell size), wh (relative to entire image) -> rescale
        coord_mask_expand = coord_mask.unsqueeze(-1).repeat(1,1,1,2) # n개의 배치에서 물체가 있는 곳만 뽑았기 때문에
        lt_norm_expand = self.lt_norm.expand_as(coord_mask_expand)  # lt_norm도 batch만큼 확장하고, 각 셀에서의 lt x,y를 뽑아줌

        # 원본 이미지 좌상 좌표{n,2} = 해당 셀의 좌상 좌표 + cxy * (셀 크기) - wh/2
        # rb좌표 = lt좌표 + wh
        iou_pred[:, :2] = lt_norm_expand[coord_mask] + iou_pred[:, :2]*(1/self.S) - iou_pred[:, 2:4]/2
        iou_pred[:, 2:4] += iou_pred[:, :2]
        iou_pred[:, 5:7] = lt_norm_expand[coord_mask] + iou_pred[:, 5:7]*(1/self.S) - iou_pred[:, 5:7]/2
        iou_pred[:, 7:9] += iou_pred[:, 5:7]
        #
        iou_target[:, :2] = lt_norm_expand[coord_mask] + iou_target[:, :2] * (1 / self.S) - iou_target[:, 2:4] / 2
        iou_target[:, 2:4] += iou_target[:, :2]  # rb좌표 = lt좌표 + wh

        # bb1, bb2's iou {n}
        iou1 = get_iou(iou_pred[:, :4],  iou_target[:, :4])
        iou2 = get_iou(iou_pred[:, 5:9], iou_target[:, :4])

        # best iou index : high_bbox {n,10}
        high_bbox = torch.cat([(iou1 >= iou2).unsqueeze(-1).expand_as(iou_target[:, :5]),
                               (iou1 < iou2).unsqueeze(-1).expand_as(iou_target[:, :5])],dim=-1)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ loss tensor3. coord에서 계산이 필요한 부분만 mat으로 분류 -> loss 계산
        # 1. bbox 예측
        mat_bbox_pred = modely[coord_mask][:, :10][high_bbox].contiguous().view(-1,5)[:,:4]
        mat_bbox_target = targety[coord_mask][:,:4]

        loss_xy = self.mse(mat_bbox_pred[:, :2],
                           mat_bbox_target[:, :2])

        loss_wh = self.mse(torch.sign(mat_bbox_pred[:,2:])*torch.sqrt(torch.abs(mat_bbox_pred[:,2:])), # 부호 유지
                           torch.sqrt(mat_bbox_target[:,2:]))

        # 2. iou 예측 : best iou를 갖는 bbox의 5번째 열(conf)이 best iou를 예측하도록
        mat_iou_pred =  iou_pred[high_bbox].contiguous().view(-1,5)[:,-1]
        mat_iou_target = torch.max(torch.stack([iou1,iou2],dim=-1),dim=-1)[0] # best iou

        loss_obj = self.mse(mat_iou_target,mat_iou_pred)

        # 3. no obj (background) 예측
        loss_noobj = self.mse(mat_noobj_pred, mat_noobj_target)

        # 4. label 예측
        mat_label_pred = modely[coord_mask][:,10:] # {n*1, 5*num_bboxes+num_classes}
        mat_label_target = targety[coord_mask][:, 10:]

        loss_label = self.mse(mat_label_pred,mat_label_target)

        loss = 5*(loss_xy + loss_wh) + loss_obj + .5 * loss_noobj + loss_label
        loss /= targety.shape[0]
        return loss

class yolo_v1(LightningModule):
    def __init__(self, out_c=512, num_classes=5, num_bboxes=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
        #
        self.conv1 = self.make_layer(  3,  64, kernel_size=7, stride=2, padding=3, pool=True) # 112
        self.conv2 = self.make_layer( 64, 256, kernel_size=3, padding=1, pool=True) #56
        self.conv3 = self.make_layer(256, 128, kernel_size=1, padding=0)
        self.conv4 = self.make_layer(128, 256, kernel_size=3, padding=1)
        self.conv5 = self.make_layer(256, 256, kernel_size=1, padding=0)
        self.conv6 = self.make_layer(256, 512, kernel_size=3, padding=1, pool=True) #28
        #
        tmp = []
        for i in range(1):
            tmp.extend(self.make_layer(512, 256, kernel_size=1, padding=0, raw=True))
            tmp.extend(self.make_layer(256, 512, kernel_size=3, padding=1, raw=True))
        self.conv7 = nn.Sequential(*tmp)
        #
        del tmp
        self.conv8 = self.make_layer(512, 512,  kernel_size=1, padding=0)
        self.conv9 = self.make_layer(512, out_c, kernel_size=3, padding=1, pool=True) #14
        #
        tmp = []
        for i in range(2):
            tmp.extend(self.make_layer(out_c, 512, kernel_size=1, padding=0, raw=True))
            tmp.extend(self.make_layer(512, out_c, kernel_size=3, padding=1, raw=True))
        self.conv10 = nn.Sequential(*tmp)
        del tmp
        self.conv11 = self.make_layer(out_c, out_c, kernel_size=3, padding=1)
        self.conv12 = self.make_layer(out_c, out_c, kernel_size=3, stride=2, padding=1) #7
        #
        # self.conv13 = self.make_layer(out_c, out_c, kernel_size=3, padding=1)
        # self.conv14 = self.make_layer(out_c, out_c, kernel_size=3, padding=1)

        # self.fc = nn.Sequential(flatten(),
        #                         nn.Linear(7*7*out_c, 4096),
        #                         nn.LeakyReLU(inplace=True),
        #                         nn.Dropout(0.5),
        #                         nn.Linear(4096, 7*7*(num_bboxes*5+num_classes)), # n,7,7,512 : n,7,7,30
        #                         nn.Sigmoid())

        self.fc = nn.Sequential(nn.Conv2d(out_c, 5*num_bboxes+num_classes, kernel_size=1, bias=False), # 15,7,7
                                nn.BatchNorm2d(5*num_bboxes+num_classes),
                                nn.LeakyReLU(),
                                flatten(), # 15*7*7
                                nn.Dropout(0.5),
                                nn.Linear((5*num_bboxes+num_classes)*7*7,(5*num_bboxes+num_classes)*7*7),
                                nn.Sigmoid(),
                                unflatten(num_bboxes,num_classes)) # 15,7,7
    #
    def make_layer(self,in_c,out_c,kernel_size,stride=1,padding=0,pool=False,raw=False):
        if (kernel_size!=1) and (in_c!=3):
            groups = in_c
        else:
            groups = 1
        cat = [nn.Conv2d(in_c,out_c,kernel_size=kernel_size,stride=stride,padding=padding,groups=groups,bias=False),
               nn.BatchNorm2d(out_c),
               nn.LeakyReLU(inplace=True)]
        if pool:
            cat.extend([nn.MaxPool2d(2)])
        if raw:
            return cat
        else:
            return nn.Sequential(*cat)
    #
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        #x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        #x = self.conv13(x)
        #x = self.conv14(x)
        x = self.fc(x)
        # x = x.view(-1, 7, 7, (5 * self.num_bboxes + self.num_classes)) # fc 버전
        x = x.permute(0,2,3,1)
        return x
    #
    def loss_f(self,modely,targety):
        f = Loss(num_bboxes=self.num_bboxes,num_classes=self.num_classes)
        return f(modely,targety)

    def init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    #
    def configure_optimizers(self):
        optimizer = opt.AdamW(self.parameters(), lr=0.001, weight_decay=0.001)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=5,
                                                   min_lr=1e-6)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}
    #
    def step(self, x):
        img,label = x['img'],x['label']
        y_hat = self(img)
        loss = self.loss_f(y_hat, label)
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
        print("\n* EPOCH %s | loss :{%6.6f}" % (self.current_epoch, avg_loss))
        return {'loss': avg_loss}

"""
밑에는 옛날버전 loss
"""
# def loss_f(modely,targety):
#         """
#         :param modely: n, 7,7, 5*b+c [{x,y,w,h,conf}*2 + c]
#         :param targety:
#         :return:
#         """
#         num_bboxes, num_classes = 2, 5
#
#         # 타겟에서 물체 있/없는 마스킹
#         coord_mask = targety[:, :, :, 4] > 0   # conf>0 : 실제 물체 있는 셀 (N, 7,7) // 4말고 9로 해도 똑같음
#         noobj_mask = targety[:, :, :, 4] == 0  # 없는 곳
#
#          # 셀 lt 좌표 구하기 : xywh를 역산해서 iou계산하기 위해 필요.
#         cell_x, cell_y = np.meshgrid(np.arange(7), np.arange(7))  # 7*7*2(x,y)
#         lt_norm_ = np.dstack([cell_x, cell_y]) * 1 / 7  # 일단 좌측 상단 기준 : 7*7*2(x,y)
#         lt_norm = torch.tensor(lt_norm_).unsqueeze(0).repeat(targety.shape[0],1,1,1).cuda() # n번만큼 반복
#
#         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ loss1. background ai, 타겟 : conf만 비교하면 됨
#         noobj_conf_mask = [5 * b -1 for b in range(1, num_bboxes + 1)]
#         noobj_pred = modely[noobj_mask][:,noobj_conf_mask]
#         noobj_target = targety[noobj_mask][:,noobj_conf_mask]
#
#         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ loss2,3 coord ai, 타겟
#         coord_pred = modely[coord_mask, :] # {n*7*7 중 물체, 15} -> {n,15}
#         coord_target = targety[coord_mask,:]
#
#         # loss2,3 coord ai/타겟의 bbox, class 구분 {N*7*7,:10}, {N*7*7,10:}
#         bbox_pred    = coord_pred[:, :5 * num_bboxes].view(-1,num_bboxes,num_classes)  # {n*7*7 중 물체, 5*num_bboxes} -> {specific cells, num_bboxes, 5}
#         label_pred   = coord_pred[:, 5 * num_bboxes:] # {n*7*7 중 물체, num_classes}
#         bbox_target  = coord_target[:, :5 * num_bboxes].view(-1,num_bboxes,num_classes) # {specific cells, 5},  B만큼 반복됐으니 하나만 True로 쓰면 안되나..?
#         label_target = coord_target[:, 5 * num_bboxes:]
#         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ !! loss4. iou : 5, 10번째에 있는 conf값 1을 target으로 사용하지 X, 대신에 두 bbox 중 더 큰 iou값을 target으로 잡음
#         # center x,y (relative to cell size), wh (relative to entire image) -> rescale
#         bbox1_pred = bbox_pred[:, 0, :4].clone() # bbox1 {,5}
#         bbox2_pred = bbox_pred[:, 1, :4].clone()
#         #
#         bbox1_pred = bbox_rescale(bbox1_pred,lt_norm[coord_mask])
#         bbox2_pred = bbox_rescale(bbox2_pred,lt_norm[coord_mask])
#         #
#         bbox1_target = bbox_target[:, 0, :4].clone() # bbox1 {,5}
#         bbox2_target = bbox_target[:, 1, :4].clone()
#         #
#         bbox1_target = bbox_rescale(bbox1_target,lt_norm[coord_mask])
#         bbox2_target = bbox_rescale(bbox2_target,lt_norm[coord_mask])
#         #
#         iou1 = get_iou(bbox1_pred, bbox1_target)
#         iou2 = get_iou(bbox2_pred, bbox2_target)
#         #
#         max_iou, max_ind = torch.stack([iou1,iou2],dim=1).max(1)
#
#         # 더 높은 iou를 갖는 박스만 loss로 계산
#         # max_ind를 N,2의 index로
#         max_iou_mask = torch.stack([max_ind==0,max_ind==1],dim=1)
#
#         # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ bbox중 큰거 다 뽑았고 나머지 loss 계산
#         """
#         bbox_pred[max_iou_mask][:,:4]   # 최종 예측 좌표  one bbox {N,4}
#         bbox_target[max_iou_mask][:,:4] # 최종 타겟 좌표  one bbox {N,4}
#         bbox_pred[max_iou_mask][:, 4]   # 최종 예측 conf   one bbox {N}
#         max_iou                         # 최종 타겟 conf : 모델이 예측한 xywh가 갖는 iou를 예측하도록 {N}
#         """
#
#         # loss1. : background                                   * 의문점 : 왜 noobj는 두 bbox를 전부 loss에 이용...?
#         f = nn.MSELoss(reduction='sum')
#         loss_noobj = f(noobj_pred,noobj_target)
#         loss_xy  = f(bbox_pred[max_iou_mask][:,:2],                 bbox_target[max_iou_mask][:,:2])
#         loss_wh  = f(torch.sqrt(bbox_pred[max_iou_mask][:,2:4]),    torch.sqrt(bbox_target[max_iou_mask][:,2:4]))
#         loss_obj = f(bbox_pred[max_iou_mask][:,4],                  max_iou)
#         loss_label=f(label_pred, label_target)
#         #
#         if loss_wh==np.NaN:
#             return 'Error'
#         loss = 5*(loss_xy + loss_wh) + loss_obj + .5 * loss_noobj + loss_label
#         loss /= targety.shape[0]
#         return loss
