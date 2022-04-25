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
from torch import optim as opt
from torch.utils.data import DataLoader,Dataset, TensorDataset
from torchvision import transforms
from torchvision import models
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
#
def visualize(img_raw, ob_text, bbox):
    img_raw = cv.rectangle(img_raw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=[0, 255, 0], thickness=2)
    img_raw = cv.putText(img_raw, ob_text, (bbox[0], bbox[1] - 10),
                         fontFace=cv.FONT_ITALIC,
                         fontScale=.5,
                         color=[0, 255, 0],
                         thickness=2)
    return img_raw
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

# path through : FPN으로 구현
class coco(Dataset):
    def __init__(self,image_list, image_size, model, key=['layer3'], trans=None, train=True, root_dir=r'D:\cv\Dataset/coco_2017/val2017/'):
        self.image_list = image_list
        self.image_size = image_size
        self.trans = trans
        self.train = train

        self.model = model
        self.key = key
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

        if self.trans:  # must do
            mat = self.trans(image=img,
                             bboxes=[bb],
                             category_ids=label)

        features = get_feature(self.model, self.key, mat['image'].unsqueeze(0))
        last_size = features[self.key[-1]].size(-1)

        bb_norm = torch.FloatTensor(mat['bboxes'][0])/self.image_size
        # need : location of true point (y,x), normed xy, real wh
        target = get_point(output_size=13, bbox=bb_norm)
        features['target'] = target
        return features

def get_feature(model,key,x):
    features = {}
    model.eval()
    with torch.no_grad():
        for i in [*model.named_children()][:-2]:
            x = i[1](x)
            if i[0] in key:
                features[i[0][-1]] = x.squeeze()
    return features
#
def get_point(output_size=56, bbox=[0.1,0.2,0.3,0.4]):
    center = bbox[:2]+bbox[2:]/2

    points = (center//(1/13)).long()
    residual = center%(1/13)
    normed = residual/(1/13)
    return points, torch.cat([normed,bbox[2:]],dim=0)
#


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