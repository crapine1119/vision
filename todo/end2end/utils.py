import numpy as np
import pandas as pd
import random
import os
import gc
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from glob import glob as glob
#
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset, TensorDataset
from torchvision import transforms
from torchvision import models
from torch import optim as opt
import torch.optim.lr_scheduler as lr_scheduler
#
from pytorch_lightning import LightningModule

import json
import cv2 as cv
from PIL import Image as pil
#
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import parse
## voc
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

# only for detections
class voc_dataset(Dataset):
    def __init__(self,fdir = r'D:\cv\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007', phase='train', several=True, trans=None ,imshow=False):
        super().__init__()
        self.fdir = fdir
        self.phase = phase
        self.trans = trans
        self.imshow = imshow

        self.trn_files = pd.read_csv(f'{fdir}\ImageSets\Main/train.txt', header=None, dtype=object).values[:, 0]
        self.val_files = pd.read_csv(f'{fdir}\ImageSets\Main/val.txt', header=None, dtype=object).values[:, 0]
        if several :
            trn_files_s, val_files_s = [],[]
            for img_fnm in self.trn_files:
                tree = parse(f'{fdir}/Annotations/{img_fnm}.xml')
                root = tree.getroot()
                if len(root.findall('object')) > 1:
                    trn_files_s.append(img_fnm)
            #
            for img_fnm in self.val_files:
                tree = parse(f'{fdir}/Annotations/{img_fnm}.xml')
                root = tree.getroot()
                if len(root.findall('object')) > 1:
                    val_files_s.append(img_fnm)
            #
            self.trn_files = np.array(trn_files_s, dtype=np.object_)
            self.val_files = np.array(val_files_s, dtype=np.object_)
            del trn_files_s, val_files_s
        #
        self.names = [os.path.split(i)[-1][:-13] for i in glob(f'{fdir}\ImageSets\Main/*_trainval*')]
        self.labels = {i:self.names[i] for i in range(len(self.names))}
        self.n2l = {self.names[i]:i for i in range(len(self.names))} # name to label

    def __len__(self):
        if self.phase=='train': return self.trn_files.shape[0]
        else : return self.val_files.shape[0]
    #
    def prepare(self,img_fnm):
        img_dir = f'{self.fdir}/JPEGImages/{img_fnm}.jpg'
        img_raw = cv.imread(img_dir,cv.IMREAD_COLOR)
        img = cv.cvtColor(img_raw,cv.COLOR_BGR2RGB)
        #
        tree = parse(f'{self.fdir}/Annotations/{img_fnm}.xml')
        root = tree.getroot()
        return img_raw, img, root
    #
    def __getitem__(self, item):
        if self.phase=='train':
            img_fnm = self.trn_files[item]
        else: img_fnm = self.val_files[item]

        img_raw, img, root = self.prepare(img_fnm)
        #
        labels,labels_count,bboxes = [],[],[]
        count=2

        # get annotations from xml
        for ob in root.findall('object'):
            ob_text = ob.find('name').text
            xml_bbox = ob.find('bndbox')
            bbox = []
            for xy in ['xmin','ymin','xmax','ymax']:
                bbox.append(int(xml_bbox.find(xy).text))

            # visualize raw image with bboxes
            if self.imshow:
                img_raw = visualize(img_raw, ob_text, bbox)
            bboxes.append(bbox)
            l = self.n2l[ob.find('name').text]
            labels.append(l)
            if l in labels_count:
                count+=1
            labels_count.append(count)

        # use albumentation
        if self.trans: # must do
            mat = self.trans(image=img,
                             bboxes=bboxes,
                             category_ids=np.array(labels))

        label_mat = bb2seg((224,224),mat['bboxes'],labels,labels_count)

        info = {'imgs': mat['image'],
                'labels': torch.LongTensor(labels),
                # 'labels_count': torch.LongTensor(labels_count),
                'labels_mat' : torch.LongTensor(label_mat),
                # 'bboxes': torch.FloatTensor(mat['bboxes'])
                }

        if self.imshow:
            info['imgs_raw'] = img_raw
        #
        return info

def visualize(img_raw, ob_text, bbox):
    img_raw = cv.rectangle(img_raw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=[0, 255, 0], thickness=2)
    img_raw = cv.putText(img_raw, ob_text, (bbox[0], bbox[1] - 10),
                         fontFace=cv.FONT_ITALIC,
                         fontScale=.5,
                         color=[0, 255, 0],
                         thickness=2)
    return img_raw
## imagenet
def bb2seg(n2l, wh=(224,224), bboxes=[], labels=[]):
    n = 224/wh[0]
    # labeled_img = torch.LongTensor([[[0]*wh[0] for _ in range(wh[1])] for _ in range(len(n2l))]) # 1000,56,56
    labeled_img = torch.FloatTensor([[0] * wh[0] for _ in range(wh[1])])  # 56,56
    labeled_count = labeled_img.clone()
    # find duplicated loc
    for bb, l in zip(bboxes, labels):
        x1, y1, x2, y2 = [*map(lambda x: int(x / n), bb)]
        # labeled_count[l, y1:y2, x1:x2] += 1
        labeled_count[y1:y2, x1:x2] += 1
    labeled_img[labeled_count>=2] = 1

    # check unique loc
    # count = [2]*len(n2l)
    for bb, l in zip(bboxes, labels):
        x1, y1, x2, y2 = [*map(lambda x: int(x / n), bb)]
        unique_loc = labeled_img!=1 # 56,56
        bb_loc = torch.zeros_like(unique_loc)
        bb_loc[y1:y2,x1:x2] = True
        ind = unique_loc&bb_loc
        if ind.sum()>0:
            labeled_img[ind] = l+2
            # count[l] += 1
        del ind
        gc.collect()
    del labeled_count
    gc.collect()
    return labeled_img

class imagenet(Dataset):
    def __init__(self,fdir = r'D:\cv\Dataset\Imagenet', trans = None, several=True, stride=4):
        super().__init__()
        self.fdir = fdir
        self.trans = trans
        self.trn_files =  glob(f'{fdir}/ILSVRC2012_img_val/*')
        self.stride = stride
        if several :
            print('get multiple bboxes image')
            trn_files_s = []
            for img_fnm in tqdm(self.trn_files):
                xml_fnm = os.path.split(f'{img_fnm[:-5]}.xml')[-1]
                xml_fnm = f'{self.fdir}/ILSVRC2012_bbox_val_v3/val/{xml_fnm}'
                tree = parse(xml_fnm)
                root = tree.getroot()
                if len(root.findall('object')) > 1:
                    trn_files_s.append(img_fnm)
            #
            self.trn_files = np.array(trn_files_s, dtype=np.object_)
            del trn_files_s
        #
        self.n2l = {i[0]:i[1]-1 for i in pd.read_csv(f'{fdir}/labels.txt',delim_whitespace=True, header=None).values}
        self.model = nn.Sequential(*[*models.resnet18(pretrained=True).children()][:-2])
    #
    def prepare(self,img_fnm):
        xml_fnm = os.path.split(f'{img_fnm[:-5]}.xml')[-1]
        xml_fnm = f'{self.fdir}/ILSVRC2012_bbox_val_v3/val/{xml_fnm}'

        img_raw = cv.imread(img_fnm,cv.IMREAD_COLOR)
        img = cv.cvtColor(img_raw,cv.COLOR_BGR2RGB)
        #
        tree = parse(xml_fnm)
        root = tree.getroot()
        return img_raw, img, root
    #
    def __len__(self):
        return len(self.trn_files)

    def __getitem__(self, item):
        img_fnm = self.trn_files[item]
        img_raw, img, root = self.prepare(img_fnm)
        labels,bboxes = [],[]
        # get annotations from xml
        for ob in root.findall('object'):
            ob_text = ob.find('name').text
            xml_bbox = ob.find('bndbox')
            bbox = []
            for xy in ['xmin','ymin','xmax','ymax']:
                bbox.append(int(xml_bbox.find(xy).text))
            bboxes.append(bbox)

            l = self.n2l[ob_text]
            labels.append(l)
        #
        if self.trans: # must do
            mat = self.trans(image=img,
                             bboxes=bboxes,
                             category_ids=np.array(labels))

        label_mat = bb2seg(self.n2l, (224//self.stride, 224//self.stride), mat['bboxes'], labels)
        labels_emb = [0]*len(self.n2l)
        # multi class
        # for i in np.unique(labels):
        #     labels_emb[i]=1
        labels_emb[labels[0]]=1

        self.model.eval()
        features = self.model(mat['image'].unsqueeze(0)).squeeze().detach()
        info = {'imgs': features,
                'labels': torch.tensor(labels_emb),
                'labels_mat': label_mat,
                # 'bboxes': torch.FloatTensor(mat['bboxes'])
                }
        return info

class get_imagenet_tensor(TensorDataset):
    def __init__(self,imgs, labels):
        super().__init__()
        self.imgs = imgs # list
        self.labels = labels # list
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, item):
        img_     = self.imgs[item]
        label_   = self.labels[item]
        img = torch.load(img_)
        label = torch.load(label_)
        return {'imgs':img,
                'labels_mat':label.long()}
## coco
def get_coco_list(ano_fnm, limit=1000, repeat=5):
    with open(ano_fnm, 'r') as f:
        temp = json.loads(''.join(f.readlines()))
    f.close()
    image_list = []
    ctg_df = pd.DataFrame(temp['categories']).reset_index()
    ctg_df['index'] = ctg_df['index'] + np.ones(len(ctg_df), dtype=np.int64)
    id2ctg = dict(ctg_df.set_index('index')['id'])
    ctg2id = dict(ctg_df.set_index('id')['index'])
    ctg2name = dict(ctg_df.set_index('id')['name'])
    count = [0]*81
    label_list = []
    for a in tqdm(temp['annotations']):
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
    if repeat>1:
        low_class = np.arange(1,5+1)[np.array(count[1:])<limit]
        add = np.array([])
        for i in low_class:
            add = np.hstack([add, np.repeat(image_list[label_list == i], repeat - 1)])
        image_list = np.hstack([image_list,add])
    return image_list, id2ctg, ctg2name

def get_feature(model,key,x):
    features = {}
    for i in [*model.named_children()][:-2]:
        x = i[1](x)
        if i[0] in key:
            features[i[0][-1]] = x.squeeze()
    return features

class coco(Dataset):
    def __init__(self,image_list, model, key=['layer3'], trans=None, train=True, root_dir=r'D:\cv\Dataset/coco_2017/val2017/'):
        self.image_list = image_list
        self.trans = trans
        self.train = train
        self.root_dir = root_dir
        # self.key = ['layer1', 'layer2', 'layer3', 'layer4']
        self.key = key
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.size_no = '1,2,3,4,5,6'.split(',')
        self.grid_size = [1,2,4,7,14,28]
        self.img_size = torch.FloatTensor([112,56,28,16,8])/224
        self.wh_norm = torch.FloatTensor([224,112,56,28,16,8,0])/224
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        info = self.image_list[item]
        id = info['image_id']
        bb = info['bbox'].astype(float).copy()
        label = info['labels'].copy()

        img = cv.imread(r'%s%012d.jpg'%(self.root_dir,id),cv.IMREAD_COLOR)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        if self.trans:  # must do
            mat = self.trans(image=img,
                             bboxes=[bb],
                             category_ids=label)

        features = get_feature(self.model, self.key, mat['image'].unsqueeze(0))
        bb_norm = torch.FloatTensor(mat['bboxes'][0])/224

        #size_ind = ((bb_norm[2:].max() // self.img_size) == 0).sum() # 0~5
        #points = get_point(self.grid_size[size_ind],size_ind,bb_norm)
        points = get_point(28, bb_norm)
        features['targets'] = (points,  # 그 크기에서 센터의 좌표
                               bb_norm,
                               torch.LongTensor(mat['category_ids']).squeeze())
        return features

def get_point(output_size=56, bbox=[0.1,0.2,0.3,0.4]):
    start = [0,1,1+4,5+16,21+49,70+196]
    center = bbox[:2]+bbox[2:]/2
    x,y = (output_size*center).int()
    output = torch.zeros((output_size,output_size))
    output[y,x]=1.0
    # return torch.LongTensor([start[ref]+y*output_size+x])
    return output.bool()

def coco_decode(bb_norm,size_no=['1']):
    raw = bb_norm.clone()
    wh_norm = torch.FloatTensor([224,112,56,28,16,8,0])/224
    ind = torch.LongTensor([*map(lambda x:int(x)-1,size_no)])

    mul = wh_norm[ind].unsqueeze(-1)
    raw[...,2:] = bb_norm[...,2:]*mul
    return raw
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
        return x.view(x.size(0),-1)
#
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    #
    def mse(self, modely, targety):
        #f = nn.MSELoss(reduction='sum')
        f = nn.MSELoss()
        return f(modely,targety)
    #
    def entropy(self, modely, targety):
        f = nn.CrossEntropyLoss()
        return f(modely,targety)
    #
    def forward(self, modely, targety):
        """
        :param modely:  56,56 {3 : background, duplicated, other classes}
        :param targety: 56,56
        :return:
        """
        bg_ind = targety==0 # 64, 56,56
        # dp_ind = targety==1
        # cl_ind = targety>1
        #
        # loss_bg = self.mse(modely[bg_ind], targety[bg_ind]) # 0을 맟춤
        # loss_dp = self.mse(modely[dp_ind], targety[dp_ind])**.5 # 1을 맞춤
        # loss_cl = self.mse(modely[cl_ind], targety[cl_ind]) # 2~1001를 맟춤
        #
        # loss = 2*loss_bg + loss_dp + loss_cl

        # crossenrtopy test
        loss_bg = self.entropy(modely.permute(0, 2, 3, 1)[bg_ind], targety[bg_ind])
        loss_cl = self.entropy(modely.permute(0, 2, 3, 1)[~bg_ind], targety[~bg_ind])
        loss = loss_bg + loss_cl*5
        return loss
#
class res(LightningModule):
    def __init__(self):
        super().__init__()
        # model = models.resnet18(pretrained=True)
        # feature_ext = [*model.children()][:-2]
        # self.feature = nn.Sequential(*feature_ext)
        # del model, feature_ext
        # gc.collect()

        self.classifier = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(512),
                                        nn.LeakyReLU(),
                                        nn.Upsample(scale_factor=2, mode='bilinear'), #14
                                        #nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1, bias=False),
                                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
                                        nn.Conv2d(512, 1024, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(1024),
                                        nn.LeakyReLU(),
                                        nn.Upsample(scale_factor=2, mode='bilinear'), #28
                                        #nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
                                        nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, groups=1024),
                                        nn.Conv2d(1024, 2048, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(2048),
                                        nn.LeakyReLU(),
                                        nn.Upsample(scale_factor=2, mode='bilinear'), #56
                                        #nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=False),
                                        nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False, groups=2048),
                                        nn.Dropout2d(0.5),
                                        nn.Conv2d(2048, 1002, kernel_size=1, bias=False))
        self.init_weights()

    def forward(self,x):
        # x = self.feature(x)
        x = self.classifier(x)
        return x.squeeze() # b,56,56

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
        optimizer = opt.AdamW(self.parameters(), lr=0.002, weight_decay=0.005)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=5,
                                                   min_lr=1e-5)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        img,label = x['imgs'],x['labels_mat']
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
        print("\n* EPOCH %s | loss :{%4.4f}" % (self.current_epoch, avg_loss))
        return {'loss': avg_loss}

##
def IouLoss(bb1,bb2):
    # x1,y1,x2,y2
    inter_x1 = torch.max(bb1[:, 0], bb2[:, 0])
    inter_y1 = torch.max(bb1[:, 1], bb2[:, 1])
    inter_x2 = torch.min(bb1[:, 2], bb2[:, 2])
    inter_y2 = torch.min(bb1[:, 3], bb2[:, 3])

    inter_x = (inter_x2 - inter_x1).clamp(min=.0)
    inter_y = (inter_y2 - inter_y1).clamp(min=.0)

    # inter_area = inter_x*inter_y
    # uni_area =
    #
    # (bb1[:, 2]-bb1[:, 0])


    bb1[:, [0, 2]].sum(dim=-1)
    bb1[:, [1, 3]].sum(dim=-1)
    #inter_x2 =
    inter_w = torch.max(bb1[:, 0], bb2[:, 0])



    inter_h = torch.min(bb1[:, 1], bb2[:, 1])

    inter_w[inter_w<0] = 0
    inter_h[inter_h<0] = 0
    inter_area = inter_w*inter_h

    bb1_area = torch.clamp(bb1,min=0)[:,0]*torch.clamp(bb1,min=0)[:,1]
    bb2_area = bb2[:, 0] * bb2[:, 1]
    iou = inter_area/(bb1_area+bb2_area-inter_area+1e-6)

    inter_x = max(bb1[:,0],)


    return (1-iou).mean()

class Loss_free(nn.Module):
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
        elif func == 'iou':
            f = IouLoss
        return f(modely,targety)

    def forward(self, modely, targety):
        """
        :param modely:  ((points) : [n,56,56,2],   (bboxes) : [n,56,56,4], (labels) : [n,56,56,80] )
        :param targety: ((points) : [n,x,y], (bboxes) : [n,x1,y1,w,h], (labels) : [n,] )
        :return:
        """
        model_p, model_bb, model_c = modely
        target_p, target_bb, target_c = targety

        # get target location index
        ind_point = torch.zeros_like(model_p[:,:,:,0]).cuda().bool()
        for e,i in enumerate(target_p):
            ind_point[e,i[1],i[0]] = True
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 1. TF
        loss_point = self.get_loss(model_p[ind_point], # 64(B), 1
                                   torch.LongTensor([1]).cuda().repeat(model_p.size(0)),
                                   func='entropy')

        loss_bg    = self.get_loss(model_p[~ind_point], # 64(B), 56*56 -1
                                   torch.LongTensor([0]).cuda().repeat(model_p[~ind_point].size(0)),
                                   func='entropy')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2. bb

        # model_xy = model_bb[ind_point][:,:2]
        model_wh = model_bb[ind_point][:,2:]

        # loss_xy = self.get_loss(model_xy,
        #                         target_bb[:,:2]+target_bb[:,2:]/2,
        #                         func='smooth')
        loss_wh = self.get_loss(torch.sign(model_wh)* (abs(model_wh)+1e-6)**.5,
                                target_bb[:,2:],
                                func='smooth')
        # loss_wh = self.get_loss(model_wh,
        #                         target_bb[:,2:],
        #                         func='iou')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 3. class
        loss_c = self.get_loss(model_c[ind_point],
                               target_c,
                               func='entropy')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 4. iou (NMS)
        ind_o2m = (torch.max(model_c,dim=-1)[1]-target_c.view(-1,1,1))==0 # target 클래스인데, 다른곳에서 나온 애들
        nms_ind = ind_o2m & (~ind_point)

        if nms_ind.sum()==0:
            loss_nms_f = self.get_loss(model_p[~ind_point],
                                       torch.LongTensor([0]).cuda().repeat(model_p[~ind_point].size(0)),
                                       func='entropy')
        else:
            loss_nms_f = self.get_loss(model_p[nms_ind],
                                       torch.LongTensor([0]).cuda().repeat(model_p[nms_ind].size(0)),
                                       func='entropy')*2
        """
        bg  : 4
        p   : 0.02
        wh  : 6*2
        c   : 12
        nms_t : 0.02
        nms_f : 4   *2.5
        """

        loss = loss_bg + loss_point + 5*loss_wh + loss_c + loss_nms_t + loss_nms_f
        return loss

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
            self.classifier = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=False, groups=in_c),
                                            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_c),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False, groups=out_c),
                                            nn.BatchNorm2d(out_c),
                                            nn.LeakyReLU(),
                                            nn.Dropout2d(0.5),
                                            nn.Conv2d(out_c,n_class, kernel_size=1,bias=False))

        def forward(self, x):
            x = self.classifier(x)
            return x

class freenet(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.block4 = elwise_block(512, hparams.out_size)
        self.block3 = elwise_block(256, hparams.out_size)
        self.block2 = elwise_block(128, hparams.out_size)
        self.block1 = elwise_block(64, hparams.out_size)
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        #
        self.classifier = conv_3x3(in_c=hparams.out_size, n_class=81)
        self.regressor = conv_3x3(in_c=hparams.out_size, n_class=4)
        self.roi = conv_3x3(in_c=hparams.out_size, n_class=2)
        #self.nms = conv_3x3(in_c=hparams.out_size, n_class=2)
        #
        self.init_weights()

    def forward(self,x):
        elwise = self.block4(x['4']) # 2048, 7,7 -> 256, 7,7
        elwise = self.upsamp(elwise) # 256, 14,14
        elwise = self.block3(x['3']) + elwise
        elwise = self.upsamp(elwise)  # 256, 28,28
        elwise = self.block2(x['2']) + elwise
        elwise = self.upsamp(elwise)  # 256, 56,56
        elwise = self.block1(x['1']) + elwise
        #
        out_points = self.roi(elwise)
        out_bboxes = self.regressor(elwise)
        out_lables = self.classifier(elwise)
        #out_nms = self.nms(elwise)
        return out_points.permute(0,2,3,1), out_bboxes.permute(0,2,3,1), out_lables.permute(0,2,3,1)#, out_nms.permute(0,2,3,1)

    def loss_f(self, modely, targety):
        f = Loss_free()
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
        optimizer = opt.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=self.hparams.lr_patience,
                                                   min_lr=self.hparams.lr*.001)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        label = x['targets']
        y_hat = self(x)
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
        print("\n* EPOCH %s | loss :{%4.4f}" % (self.current_epoch, avg_loss))
        return {'loss': avg_loss}
##

class Loss_mask(nn.Module):
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
        elif func == 'iou':
            f = IouLoss
        return f(modely,targety)

    def forward(self, modely, targety):
        """
        :param modely:  batch, 5(random point), 85(bb4 + class81)
        :param targety: ((points) : [n,x,y], (bboxes) : [n,x1,y1,w,h], (labels) : [n,] )
        :return:
        """
        #model_p = modely
        ind_p, _, _ = targety

        pred_p = modely[1][ind_p] # 64
        pred_bg = modely[1][~ind_p] # 64

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 1. True bb
        # loss_wh = self.get_loss(torch.sign(model_wh)* (abs(model_wh)+1e-6)**.5,
        #                         target_bb[:,2:],
        #                         func='smooth')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 3. mask
        loss_p = self.get_loss(pred_p,
                               torch.ones(pred_p.size(0),dtype=torch.long).cuda(),
                               func='entropy')
        loss_bg = self.get_loss(pred_bg,
                                torch.zeros(pred_bg.size(0), dtype=torch.long).cuda(),
                                func='entropy')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 4. iou (NMS)

        loss = loss_p + .5*loss_bg
        return loss
        # return loss, {'bg':loss_bg,
        #               'p':loss_point,
        #               'xy':loss_xy,
        #               'wy':loss_wh,
        #               'c':loss_c}

class freenet3_1(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.block4 = elwise_block(512, hparams.out_size) # 7
        self.block3 = elwise_block(256, hparams.out_size) # 14
        self.block2 = elwise_block(128, hparams.out_size) # 28
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(128,2, kernel_size=1, bias=False)
        #
        self.init_weights()

    def forward(self,x):
        elwise = self.block4(x['4'])
        elwise = self.upsamp(elwise)
        elwise = self.block3(x['3']) + elwise # elwise = 128, 14,14
        elwise = self.upsamp(elwise)
        elwise = self.block2(x['2']) + elwise # elwise = 128, 28,28
        out28 = self.conv(elwise)
        #
        return elwise,out28.permute(0,2,3,1)

    def loss_f(self, modely, targety):
        f = Loss_mask()
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
        optimizer = opt.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=self.hparams.lr_patience,
                                                   min_lr=self.hparams.lr*.001)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        label = x['targets']
        y_hat = self(x)
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
        print("\n* EPOCH %s | loss :{%4.4f}" % (self.current_epoch, avg_loss))
        return {'loss': avg_loss}

class bridge(TensorDataset):
    def __init__(self, dataset, model):
        super().__init__()
        self.dataset = dataset
        self.model = model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x = self.dataset[item]
        for i in ['4','3','2']:
            x[i] = x[i].unsqueeze(0)
        target = x['targets']
        target_mask,target_bb,target_c = target
        target_n = (target_mask.unsqueeze(0),target_bb.unsqueeze(0),target_c.unsqueeze(0))
        x['targets'] = target_n
        self.model.eval()
        with torch.no_grad():
            feature,mask = self.model(x) # 128,28,28 // 1,28,28

        score,ind = torch.max(nn.Softmax(dim=-1)(mask.squeeze()),dim=-1) # 28,28
        score.size(), ind.size()
        score[ind==0]=0.0
        best = torch.argsort(score.flatten(), descending=True)[:100]
        mask_best = torch.zeros_like(ind)
        for b in best:
            mask_best[b//28,b%28] = 1.0
        if (target_mask & mask_best.bool()).sum()==0:
            target_false = torch.zeros_like(ind)
            target_false[best[-1]//28,best[-1]%28] = 1.0
            return {'feature'   : feature.squeeze()[:,mask_best.bool()], # 100
                    'target'    : (target_false[mask_best.bool()],torch.FloatTensor([0.0,0.0,0.0,0.0]),torch.zeros((1,)).long())}

        return {'feature'   : feature.squeeze()[:,mask_best.bool()], # 100
                'target'    : (target_mask[mask_best.bool()],target_bb,target_c.unsqueeze(0))}

class Loss_last(nn.Module):
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
        elif func == 'iou':
            f = IouLoss
        return f(modely,targety)

    def forward(self, modely, targety):
        """
        :param modely:  b, 10, 100
        :param targety: p(b,100), bb(b,4), c(b,1+cls)
        :return:
        """

        ind_p, target_bb, target_c = targety
        ind_p = ind_p.bool()
        pred_bb = modely[ind_p][:,:4] # b, 4
        pred_bg = modely[~ind_p][:,4:] # b*99, 6
        pred_c = modely[ind_p][:,4:]  # b, 6
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 1. True bb
        loss_bb = self.get_loss(pred_bb,
                                target_bb,
                                func='smooth')
        # loss_wh = self.get_loss(torch.sign(model_wh)* (abs(model_wh)+1e-6)**.5,
        #                         target_bb[:,2:],
        #                         func='smooth')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2. bg, class
        loss_bg = self.get_loss(pred_bg,
                                torch.zeros(pred_bg.size(0), dtype=torch.long).cuda(),
                                func='entropy')

        loss_c = self.get_loss(pred_c,
                               target_c.squeeze(),
                               func='entropy')

        loss = loss_bb + .1*loss_bg + loss_c
        return loss
        # return loss, {'bg':loss_bg,
        #               'p':loss_point,
        #               'xy':loss_xy,
        #               'wy':loss_wh,
        #               'c':loss_c}

class freenet3_2(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.conv1 = nn.Sequential(nn.Conv1d(128,10, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(10),
                                   nn.ReLU())
        self.fc = nn.Sequential(nn.Dropout(0.7),
                                nn.Linear(1000,1000, bias=True))
        #
        self.init_weights()

    def forward(self,x):
        output = self.conv1(x['feature']) # b,10b100
        output = output.view(-1,1000)
        output = self.fc(output)
        return output.view(-1,10,100).permute(0,2,1)

    def loss_f(self, modely, targety):
        f = Loss_last()
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
        optimizer = opt.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=self.hparams.lr_patience,
                                                   min_lr=self.hparams.lr*.001)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        label = x['target']
        y_hat = self(x)
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
        print("\n* EPOCH %s | loss :{%4.4f}" % (self.current_epoch, avg_loss))
        return {'loss': avg_loss}