import numpy as np
import pandas as pd
import random
import os
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
#
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import parse
## setting
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
        self.several = several
        self.trans = trans
        self.imshow = imshow

        self.trn_files = pd.read_csv(f'{fdir}\ImageSets\Main/train.txt', header=None, dtype=object).values[:, 0]
        self.val_files = pd.read_csv(f'{fdir}\ImageSets\Main/val.txt', header=None, dtype=object).values[:, 0]
        if self.several :
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
        return img_dir, img_raw, img, root
    #
    def visualize(self,img_raw, ob_text, bbox):
        img_raw = cv.rectangle(img_raw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=[0, 255, 0], thickness=2)
        img_raw = cv.putText(img_raw, ob_text, (bbox[0], bbox[1] - 10),
                             fontFace=cv.FONT_ITALIC,
                             fontScale=.5,
                             color=[0, 255, 0],
                             thickness=2)
        return img_raw
    #
    def __getitem__(self, item):
        if self.phase=='train':
            img_fnm = self.trn_files[item]
        else: img_fnm = self.val_files[item]

        img_dir, img_raw, img, root = self.prepare(img_fnm)
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
                img_raw = self.visualize(img_raw, ob_text, bbox)
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

        info = {'imgs': mat['image'],
                'labels': torch.LongTensor(labels),
                'labels_count': torch.LongTensor(labels_count),
                'bboxes': torch.FloatTensor(mat['bboxes'])}

        if self.imshow:
            info['imgs_raw'] = img_raw
        #
        return info

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