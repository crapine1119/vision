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

def get_items(ano_fnm):
    with open(ano_fnm, 'r') as f:
        temp = json.loads(''.join(f.readlines()))
    f.close()
    image_list = []
    ctg_df = pd.DataFrame(temp['categories']).reset_index()
    ctg_df['index'] = ctg_df['index'] + np.ones(len(ctg_df), dtype=np.int64)
    id2ctg = dict(ctg_df.set_index('index')['id'])
    ctg2id = dict(ctg_df.set_index('id')['index'])
    ctg2name = dict(ctg_df.set_index('id')['name'])

    for a in temp['annotations']:
        image_id = a['image_id']
        bbox = np.stack(a['bbox'])
        #labels = np.asarray([ctg2id[l] for l in a['category_id']])
        labels = np.asarray([ctg2id[a['category_id']]])
        image_list.append({'image_id':image_id, 'bbox':bbox, 'labels':labels})
    return np.asarray(image_list), id2ctg, ctg2name

def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

## customizing
import os

class customdataset(Dataset):
    def __init__(self,sdir,ctg2id,transform=None):
        self.samples = glob.glob('%s/*.jpg'%sdir)

        # load all (car) images
        # self.jpeg_images = [torch.tensor(cv.imread(sample_name),dtype=torch.float32) for sample_name in self.samples]
        # positive : iou >= 0.5 negative : iou < 0.5
        # Save positive and negative separately
        self.ctg2id = ctg2id
        self.positive_annotations = [glob.glob(r'%s/*%s*p.csv' % (sdir, os.path.split(sample_name)[-1][:12]))[0]
                                     for sample_name in self.samples]
        self.negative_annotations = [glob.glob(r'%s/*%s*n.csv' % (sdir, os.path.split(sample_name)[-1][:12]))[0]
                                     for sample_name in self.samples]

        self.labels = [int(os.path.split(i)[-1][13:-6]) for i in self.positive_annotations]

        # bounding box sizes
        self.grt = [torch.tensor(pd.read_csv('%s_grt.csv'%i[:-4]).values[0],dtype=torch.float32) for i in self.samples]
        self.positive_rects, self.negative_rects = [], []  # positive_rects = [(x, y, w, h), ....]
        self.positive_sizes, self.negative_sizes = [], [] # positive_sizes = [1, .....]

        # bounding box coordinates
        for annotation_path in self.positive_annotations:
            rects = pd.read_csv(annotation_path).values
            # The existing file is empty or there is only a single line of data in the file
            if len(rects) >0:
                try:
                    self.positive_rects.append(torch.tensor(rects,dtype=torch.int))
                    self.positive_sizes.append(len(rects))
                except:
                    self.positive_rects.append(torch.tensor([]))
                    self.positive_sizes.append(0)
        #
        for annotation_path in self.negative_annotations:
            rects = pd.read_csv(annotation_path).values[:self.positive_sizes[0]]
            if len(rects) >0:
                try:
                    self.negative_rects.append(torch.tensor(rects,dtype=torch.int))
                    self.negative_sizes.append(len(rects))
                except:
                    self.negative_rects.append(torch.tensor([]))
                    self.negative_sizes.append(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,i):
        self.p_img = []

        tmp = cv.imread(self.samples[i])
        for p in self.positive_rects[i]:
            if len(p)>0:
                roi = p.numpy()
                resized = tmp[roi[1]:roi[3],roi[0]:roi[2]]
                resized = cv.resize(resized, (224,224),interpolation = cv.INTER_AREA)
                resized = torch.tensor(resized,dtype=torch.float32).permute(-1,0,1)/255
                self.p_img.append(resized)
        self.n_img = []
        for n in self.negative_rects[i]:
            if len(n) > 0:
                roi = n.numpy()
                resized = tmp[roi[1]:roi[3],roi[0]:roi[2]]
                resized = cv.resize(resized, (224,224),interpolation = cv.INTER_AREA)
                resized = torch.tensor(resized,dtype=torch.float32).permute(-1,0,1)/255
                self.n_img.append(resized)

        imgs = torch.cat([torch.stack(self.p_img),torch.stack(self.n_img)],dim=0) # 15, 224, 224, 3

        onehot = torch.zeros((len(imgs)),dtype=torch.long)
        #onehot[:len(self.p_img)] = self.ctg2id[self.labels[i]]
        onehot[:len(self.p_img)] = 1

        return {'img': imgs,
                'label': onehot,
                'grt': self.grt[i],
                'p_rect':self.positive_rects[i],
                'p_size':self.positive_sizes[i],
                'n_rect':self.negative_rects[i],
                'n_size':self.negative_sizes[i]}

##
def calcul(positive,grt):
    xmin, ymin, xmax, ymax = positive[:,0],positive[:,1],positive[:,2],positive[:,3]
    p_w = xmax - xmin
    p_h = ymax - ymin
    p_x = xmin + p_w / 2
    p_y = ymin + p_h / 2
    #
    xmin, ymin, xmax, ymax = grt
    g_w = xmax - xmin
    g_h = ymax - ymin
    g_x = xmin + g_w / 2
    g_y = ymin + g_h / 2
    #
    t_x = (g_x - p_x) / p_w
    t_y = (g_y - p_y) / p_h
    t_w = torch.log(g_w / p_w)
    t_h = torch.log(g_h / p_h)
    t_target = torch.cat([t_x.unsqueeze(1),t_y.unsqueeze(1),t_w.unsqueeze(1),t_h.unsqueeze(1)],dim=1)
    return t_target