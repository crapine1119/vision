import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm as tqdm
from glob import glob as glob
#
import torch
from torch import nn
from torch import optim as opt
from torch.utils.data import DataLoader,Dataset, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
#
from pytorch_lightning import LightningModule

import json
import cv2 as cv
from PIL import Image as pil
from torchvision.ops import sigmoid_focal_loss as focal_loss
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase as cbar_base
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

# coco
def get_coco_list(ano_fnm, limit=1000, repeat=5, num_class=10):
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
        if (labels<=num_class) and (count[labels.item()]<limit):
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
    frac = [*model.named_children()][:-2]
    for i in frac:
        with torch.no_grad():
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

        points,targets = get_point(28, bb_norm, mat['category_ids'][0], mask=True, segment=True)
        features['targets'] = (points,  # 그 크기에서 센터의 좌표
                               bb_norm,
                               targets)
                               # torch.LongTensor(mat['category_ids']).squeeze())
        return features

def get_point(output_size=56, bbox=[0.1,0.2,0.3,0.4], label=0, mask=True, segment=True):
    center = bbox[:2]+bbox[2:]/2
    x,y = (output_size*center).int()
    x1,y1 = (output_size* bbox[:2]).int()
    x2,y2 = (output_size * (bbox[:2]+bbox[2:])).int()
    if mask:
        output = torch.zeros((output_size,output_size,2))
        target = torch.zeros((output_size, output_size))
        if segment:
            output[y1:y2+1, x1:x2+1, 1] = 1.0
            output[y,x,0] = .01 # center
            target[y1:y2+1, x1:x2+1] = label
        else:
            output[y,x,1]=1.0
        return output, target.long()
    #return torch.LongTensor([x,y])
    # return torch.LongTensor([start[ref]+y*output_size+x])

## basic modules
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

## model
class Loss_mask2(nn.Module):
    def __init__(self, alpha=0.25, gamma = 2):
        super().__init__()
        # self.alpha = alpha
        # self.gamma = gamma
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
        :param modely : {b, 28,28, 6}
        :param targety: (mask, bb, label) {b,28,28,x}
        :return:
        """
        model_c,model_bb = modely
        target_p, target_bb, target_c = targety
        """
        model_p   : b, 28,28,2
        target_p  : b, 28,28,2
        model_c   : b, 28,28,5
        target_c  : b
        model_bb  : b, 28,28,2 (prob, w,h)
        target_bb : b,4 (x1,y1,w,h)  
        """

        _, ind_p = torch.max(target_p,dim=-1)
        loss_c = self.get_loss(model_c[ind_p.detach().bool()],
                               target_c[ind_p.detach().bool()],
                               func='entropy')

        unknown = ~(ind_p.detach().bool())
        loss_un = self.get_loss(model_c[unknown],
                                torch.zeros(unknown.sum()).cuda().long(),
                                func='entropy')

        ind_center = target_p[...,0]==.01
        loss_bb = self.get_loss(model_bb[ind_center.detach().bool()],
                                target_bb[...,2:],
                                func='mse')
        # 잘못된 point에서 C를 검출했을 경우의 페널티가 필요함 : Add Center loss (uclidean distance)

        loss = loss_c + loss_un + loss_bb
        return loss

class toy2(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.block4 = elwise_block(512, hparams.out_size) # 7
        #
        self.block3 = elwise_block(256, hparams.out_size) # 14
        self.conv3 = nn.Sequential(nn.Conv2d(hparams.out_size,hparams.out_size,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(hparams.out_size),
                                   nn.ReLU())
        #
        self.block2 = elwise_block(128, hparams.out_size) # 28
        self.conv2 = nn.Sequential(nn.Conv2d(hparams.out_size,hparams.out_size,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(hparams.out_size),
                                   nn.ReLU())
        #
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')

        self.cls = nn.Conv2d(hparams.out_size, self.hparams.num_class+1, kernel_size=1, bias=False)
        # added for regression
        self.reg = nn.Sequential(nn.BatchNorm2d(self.hparams.num_class+1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.hparams.num_class+1, self.hparams.num_class+1, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(self.hparams.num_class + 1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.hparams.num_class + 1, self.hparams.num_class + 1, kernel_size=1,bias=False),
                                 nn.BatchNorm2d(self.hparams.num_class + 1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.hparams.num_class+1, 2, kernel_size=1, bias=False),
                                 nn.ReLU()) # prob + wh
        #
        self.init_weights()

    def forward(self,x):
        elwise = self.block4(x['4'])
        elwise = self.upsamp(elwise)
        #
        elwise = self.block3(x['3']) + elwise # elwise = 128, 14,14
        elwise = self.conv3(elwise)
        elwise = self.upsamp(elwise)
        #
        elwise = self.block2(x['2']) + elwise # elwise = 128, 28,28
        elwise = self.conv2(elwise)

        out_c = self.cls(elwise)  # n,5,28,28
        out_r = self.reg(out_c)
        #
        return out_c.permute(0,2,3,1), out_r.permute(0,2,3,1)

    def loss_f(self, modely, targety):
        f = Loss_mask2(alpha=self.hparams.alpha, gamma=self.hparams.gamma)
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

## visualize
def visualize(model, dataset, label_list, key, id2name, save, save_dir, resize_n=672 ,num_class=4, pick=1, top_n=.0, c_threshold=.4, root_dir = r'D:\cv\Dataset/coco_2017/val2017/'):
    jet = plt.get_cmap('Set3', num_class+1)
    my_cmap = jet.colors[:num_class+1,:3]
    my_cmap[0] = [1,1,1]

    max_n = len(label_list)
    # pick randomly
    for n in np.random.randint(0, max_n, pick):
##
        dset = dataset[n]
        info = label_list[n]
        for k in key:
            dset[k[-1]] = dset[k[-1]].unsqueeze(0)
        target_p,target_bb,target_c = dset['targets']

        # predict
        model.eval()
        with torch.no_grad():
            output = model(dset)
        output_c,output_bb = output     # output_p, output_c = output # toy1

        # classification
        out_score_c, out_c = torch.max(nn.Softmax(dim=-1)(output_c[0]), dim=-1)
        pred_score = out_score_c.clone()
        pred_score[out_c==0] = .0

        # drop under top_n | threshold
        top_ind = torch.zeros_like(pred_score.flatten())
        top_ind[torch.argsort(pred_score.flatten(),descending=True)[:top_n]] = 1.0
        top_ind = top_ind.view(pred_score.size()).bool()
        pred_c = out_c.clone().float()
        if top_n:
            pred_c[~top_ind]=.0
            pred_score[~top_ind]=.0
        else:
            pred_c[out_score_c < c_threshold] = .0
            pred_score[out_score_c < c_threshold] = .0

        # raw img
        img_raw = cv.imread(r'%s/%012d.jpg' % (root_dir, info['image_id']))
        img = cv.resize(img_raw,(resize_n,resize_n))/255
        h,w = resize_n,resize_n

        # upsample heatmap
        pred_up = nn.Upsample(scale_factor=h/pred_c.size(0),mode='nearest')(pred_c.unsqueeze(0).unsqueeze(0).type(torch.float64))[0,0]
        hm = torch.zeros_like(torch.tensor(img))
        for i in range(1, num_class+1):
            hm[pred_up == i] = torch.tensor([my_cmap[i,::-1]], dtype=torch.float64).repeat((pred_up == i).sum(), 1) # b : person
        img_h = cv.addWeighted(hm.numpy(),1.0,img,0.4,0)
        img_h = np.clip(img_h, a_min=0, a_max=1.0)

        # heatmap by score
        pred_score_up = nn.Upsample(scale_factor=h / pred_score.size(0), mode='bilinear')(pred_score.unsqueeze(0).unsqueeze(0).type(torch.float64))[0, 0]

        # grt
        grt = (target_bb*resize_n).int().numpy() ############################################################
        img_h = cv.rectangle(img_h, grt[:2],grt[:2]+grt[2:],color=[0,0,1],thickness=2)
        img_h = cv.putText(img_h, id2name[info['labels'][0]], grt[:2]-np.array([0,20]), fontFace=cv.FONT_ITALIC, fontScale=0.8, color=[0, 0, 1], thickness=2)

        # predicted box (ROI)
        xx,yy = np.meshgrid(range(resize_n),range(resize_n))
        for i in range(1,num_class+1):
            ind_c = pred_up == i

            if not ind_c.sum()==0:
                score = pred_score_up[ind_c].mean().item()

                center = (pred_score * (pred_c==i)).argmax()
                cx,cy = center%28,center//28

                pred_bb = output_bb.squeeze()[cy,cx].clip(min=0).numpy()
                if (pred_bb==0).sum()==0:
                    # bb by regress
                    cxy = np.array([cx * resize_n / 28, cy * resize_n / 28])
                    bb1 = np.clip(cxy - (pred_bb / 2 * resize_n), a_min=0, a_max=resize_n).astype(int)
                    bb2 = np.clip(cxy + (pred_bb / 2 * resize_n), a_min=0, a_max=resize_n).astype(int)


                    img = cv.rectangle(img, bb1, bb2, color=[0, 1, 0], thickness=2)
                    img = cv.putText(img, '%s' % (id2name[i]), (bb1[0], bb1[1] - 20), fontFace=cv.FONT_ITALIC,
                                     fontScale=1.0, color=[0, 1, 0], thickness=2)
                else:print('Regress Fail',pred_bb)
                # bb by segment
                # roi_x = xx[ind_c]
                # roi_y = yy[ind_c]
                # x1, y1 = max(roi_x.min(),1), max(1,roi_y.min())
                # x2, y2 = min(roi_x.max(),resize_n-2), min(roi_y.max(),resize_n-2)
                #
                # img_h = cv.rectangle(img_h, (x1,y1), (x2,y2), color=[0, 1, 0], thickness=1)
                # img_h = cv.putText(img_h, '%s : %2.3f'%(id2name[i],score), (x1, y1-20), fontFace=cv.FONT_ITALIC,
                #                    fontScale=0.8, color=[0, 1, 0], thickness=2)

        # imshow & write
        #cv.imshow('%s_%s' % (id2name[info['labels'][0]], n), img_h)
        #cv.imwrite(r'D:\cv\toy\naverlabs\log\version_0\test/%s_%s.jpg' % (id2name[info['labels'][0]], n),img_h*255)
        #cv.destroyAllWindows()

        fig =  plt.figure(figsize=[17,8.5])
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img_h[...,::-1])
        # ax1.imshow(cv.resize(img_raw,(resize_n,resize_n))[...,::-1])
        #3
        ax2.imshow(img[...,::-1])
        # ax2.imshow(pred_score_up.numpy(), alpha=0.6, cmap='Reds')

        if (pred_score_up.sum()>0) & (pred_score_up.max()>c_threshold):
            ax2.contourf(pred_score_up.numpy(),
                         alpha=0.2,
                         levels=np.arange(c_threshold*.75,
                                          pred_score_up.numpy().max(),
                                          (pred_score_up.numpy().max()-c_threshold*.75)/20),
                         cmap='jet')
        #
        cax = fig.add_axes([.125,.072,.775,.03])
        cbar = cbar_base(cax, orientation='horizontal', cmap=jet)
        cbar.set_ticks(np.arange(.1,1,.2))
        cbar.set_ticklabels([id2name[i] for i in range(5)])
        cbar.ax.tick_params(labelsize=20)

        fig.suptitle('ROI : %s_%s' % (id2name[info['labels'][0]], n))
        if save:
            fig.savefig(r'%s/%s_%s.jpg' % (save_dir,id2name[info['labels'][0]], n),dpi=200,bbox_inches='tight')
            plt.close()