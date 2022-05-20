import os
import random
import numpy as np
import pandas as pd
from glob import glob as glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image as pil
import cv2 as cv
# torch
import torch
import torch.nn as nn
from torch import optim as opt
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset,DataLoader
from torchvision import models
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, accuracy_score
import timm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
## preprocess
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

def split(rdir, rate=0.2):

    train_png = sorted(glob(f'{rdir}/train/*'))
    test_png = sorted(glob(f'{rdir}/train/*'))
    test_png = sorted(glob(f'{rdir}/test/*'))
    tools = {}

    # label (0~88)
    train_y_ = pd.read_csv(rdir + "/train_df.csv")
    train_y = train_y_['label']
    label_unique = sorted(np.unique(train_y))
    tools['cs2i'] = {c: i for i, c in enumerate(label_unique)}
    tools['i2cs'] = {i: c for i, c in enumerate(label_unique)}
    train_labels = [tools['cs2i'][k] for k in train_y]

    # cls, sts
    classes = sorted(np.unique(train_y_['class']))
    states = sorted(np.unique(train_y_['state']))

    tools['c2i'] = {c: i for i, c in enumerate(classes)}
    tools['i2c'] = {i: c for i, c in enumerate(classes)}
    tools['s2i'] = {c: i for i, c in enumerate(states)}
    tools['i2s'] = {i: c for i, c in enumerate(states)}

    def reverse(x):
        return [tools['c2i'][x.split('-')[0]], tools['s2i'][x.split('-')[1]]]
    cs2cs = {i:reverse(c)  for i, c in enumerate(label_unique)}

    # data split
    trnx, valx, trny, valy = train_test_split(train_png, train_labels, test_size=rate)
    return trnx, valx, trny, valy, tools, test_png

def oversample(trnx_,trny_, min_num=20):
    """
    :param trnx_: list fnms
    :param trny_: list int
    :param min_num:
    :return:
    """
    numbers = [0]*88
    for i in trny_:
        numbers[i]+=1
    #plt.bar(range(88),numbers)

    add_x = []
    add_y = []
    for i in range(88):
        ind = np.array(trny_)==i
        c_num = ind.sum()
        if c_num<min_num:
            a,b = min_num//c_num-1, (min_num%c_num)/c_num
            if a!=0:
                cat_x = np.array(trnx_)[ind].repeat(a).tolist()
                cat_y = [i]*len(cat_x)
                add_x.extend(cat_x)
                add_y.extend(cat_y)
            if b!=0:
                _,cat_x,_,cat_y = train_test_split(np.array(trnx_)[ind].tolist(),[i]*c_num,test_size=b)
                add_x.extend(cat_x)
                add_y.extend(cat_y)
    print(len(add_y),'data is added')


    mat = np.array([add_x,add_y])

    ind_shf = np.arange(mat.shape[-1])
    np.random.shuffle(ind_shf)

    trnx = trnx_ + mat[0,ind_shf].tolist()
    trny = trny_ + mat[1,ind_shf].astype(int).tolist()

    # numbers_aug = [0]*88
    # for i in trny:
    #     numbers_aug[i]+=1

    return trnx,trny

class custom_dset(Dataset):
    def __init__(self,  dset_x, dset_y=None, trans=None, train=True, resize=256):
        super().__init__()
        self.dset_x = dset_x
        self.dset_y = dset_y
        self.trans = trans
        self.train = train
        self.resize = resize

        self.trans_test = A.Compose([A.Resize(self.resize, self.resize, interpolation=cv.INTER_AREA),
                                     A.Normalize(),
                                     ToTensorV2()])

    def __len__(self):
        return len(self.dset_x)

    def __getitem__(self, item):
        mat = {}
        img = cv.imread(f'{self.dset_x[item]}',cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        c = self.dset_y[item]
        if self.train:
            if self.trans:
                img = self.trans(image=img)['image']
        else: # val,test
            img = self.trans_test(image=img)['image']

        mat['img'] = img
        mat['label_c'] = torch.LongTensor([c]).squeeze()
        return mat

def get_callbacks(hparams):
    log_path = '%s'%(hparams.sdir)
    tube     = Tube(name=hparams.name, save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='{epoch:02d}-{val_loss:.4f}',
                                          save_top_k=1,
                                          mode='min')
    early_stopping        = EarlyStopping(monitor='val_loss',
                                          patience=hparams.es_patience,
                                          verbose=True,
                                          mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    print('get (1)ckp, (2)es, (3)lr_monitor callbacks with (4)tube')
    return {'callbacks':[checkpoint_callback, early_stopping, lr_monitor],
            'tube':tube}

class net(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        #self.pretrain = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=88)
        self.pretrain = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=88, drop_rate=0.2) # v2
        #self.pretrain = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=88, drop_rate = 0.5) #v3

        self.result = []
    def forward(self,x):
        out_c = self.pretrain(x['img'])
        return out_c

    def loss_f(self, modely, targety):
        f = nn.CrossEntropyLoss()
        return f(modely, targety)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=self.hparams.lr_patience,
                                                   min_lr=self.hparams.lr*.0001)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        label_c = x['label_c']
        y_hat = self(x)
        loss = self.loss_f(y_hat, label_c)
        pred_c = y_hat.argmax(-1).detach().cpu().tolist()
        f1_c = score_function(label_c.cpu().tolist(), pred_c)
        return loss, f1_c

    def training_step(self, batch, batch_idx):
        loss,f1_c = self.step(batch)
        self.log('trn_loss', loss, on_step=False, on_epoch=True)
        self.log('trn_f1',   f1_c, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss,f1_c = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_f1',   f1_c, on_step=False, on_epoch=True)
        return {'val_loss': loss,'f1':f1_c}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([op['val_loss'] for op in outputs]).mean()
        avg_f1 = torch.stack([op['f1'] for op in outputs]).mean()

        print("\n* EPOCH %s | loss :{%4.4f} | f1 :{%2.2f}" % (self.current_epoch, avg_loss, avg_f1))
        return {'loss': avg_loss}

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        pred_c = y_hat.argmax(-1).detach().cpu().tolist()
        self.result.extend(pred_c)











##
# extract numbers of state in each class,  oversampling with referenced number
def oversample_(zero_num, nonzero_num, rdir, classes, states,      trnx_,trny1_,trny2_,        valx_,valy1_,valy2_,tstx_,        cs2i,c2i, labeling='flat'):
    trnx = []
    trny = []
    mat = {}
    valx = np.array(r'%s/train/' % (rdir) + valx_)
    tstx = np.array(r'%s/test/' % (rdir) + tstx_)


    # single estimates
    if labeling=='flat':
        numbers,_ = np.histogram(trny1_, np.arange(88 + 1) - .5)
        numbers_aug = np.zeros_like(numbers)
        c_good = [cs2i[i] for i in cs2i if i.split('-')[-1] == 'good']
        for c in np.arange(88):
            ind = trny1_==c
            tmp_n = numbers[c]
            if c in c_good:
                ref_num = int(zero_num)
            else:
                ref_num = int(nonzero_num)

            if tmp_n <= ref_num:  # and (s!=0):
                cat_x = r'%s/train/' % (rdir) + trnx_[ind].repeat(int(ref_num / tmp_n))
            else:
                cat_x,_ = train_test_split(r'%s/train/' % (rdir) + trnx_[ind], train_size=ref_num / tmp_n)
            cat_y = trny1_.loc[cat_x.index]
            #
            trnx.extend(cat_x.values.tolist())
            trny.extend(cat_y.values.tolist())
            numbers_aug[c] = len(cat_y)
        valy = np.array(valy1_)
    #multiple estimates
    else:
        numbers = [[0]*len(states) for _ in range(len(classes))] # class, state {15,49}
        numbers = np.array(numbers)
        numbers_aug = numbers.copy()

        for c in [c2i[i] for i in classes]:
            state_c = np.sort(trny2_['state'][trny2_['class'] == c].unique())
            bins = np.hstack([state_c-.5,state_c[-1]+.5])
            #
            nums,_=np.histogram(trny2_['state'][trny2_['class']==c],bins)
            numbers[c][state_c] = nums
            #
            for s in state_c:
                tmp_n = numbers[c][s]
                ind = ((trny2_['class'] == c) & (trny2_['state'] == s))
                if s==0:
                    ref_num = int(zero_num)
                else:
                    ref_num = int(nonzero_num)

                if tmp_n <= ref_num:  # and (s!=0):
                    cat_x = r'%s/train/' % (rdir) + trnx_[ind].repeat(int(ref_num / tmp_n))
                else:
                    cat_x, _ = train_test_split(r'%s/train/' % (rdir) + trnx_[ind], train_size=ref_num / tmp_n)

                cat_y = trny2_.loc[cat_x.index]
                #
                trnx.extend(cat_x.values.tolist())
                trny.extend(cat_y.values.tolist())
                numbers_aug[c][s] = len(cat_y)
        valy = np.array(valy2_)

    trnx = np.array(trnx)
    trny = np.array(trny)

    mat['train'] = (trnx, trny)
    mat['valid'] = (valx, valy)
    mat['test'] = tstx
    mat['numbers'] = numbers
    mat['numbers_aug'] = numbers_aug
    return mat

# configure by figure
def confine(classes, states, numbers, numbers_aug, i2c, labeling='flat'):
    # class, state별 분포 재확인
    if labeling=='flat':
        fig = plt.figure(figsize=[18, 8.5])
        ax1 = fig.add_subplot(121)
        ax1.bar(range(len(numbers)), numbers)
        #ax1.set_xticks(range(len(numbers)))
        ax1.grid('on')
        ax1.set_title('Before')

        ax2 = fig.add_subplot(122)
        ax2.bar(range(len(numbers_aug)), numbers_aug)
        #ax2.set_xticks(range(len(numbers_aug)))
        ax2.grid('on')
        ax2.set_title('After')

    else:
        # 1. before Aug
        fig = plt.figure(figsize=[18, 8.5])
        ax1 = fig.add_subplot(121)
        ax1.bar(range(len(classes)), numbers.sum(axis=1))
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=30, fontsize=8.5)
        ax1.grid('on')
        #
        ax2 = fig.add_subplot(122)
        ax2.bar(range(len(states)), numbers.sum(axis=0))
        ax2.grid('on')
        ax2.set_title('states')
        fig.suptitle('Before')

        # 2. after
        fig = plt.figure(figsize=[18, 8.5])
        ax1 = fig.add_subplot(121)
        ax1.bar(range(len(classes)),numbers_aug.sum(axis=1))
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes,rotation=30,fontsize=8.5)
        ax1.grid('on')
        #
        ax2 = fig.add_subplot(122)
        ax2.bar(range(len(states)),numbers_aug.sum(axis=0))
        ax2.grid('on')
        ax2.set_title('states')
        fig.suptitle('After')

        # 3. both class & state
        fig = plt.figure(figsize=[18,8.5])
        for e,info in enumerate(numbers_aug):
            ax = fig.add_subplot(5,3,e+1)
            xtick = range((info != 0).sum())

            ax.bar(xtick, info[info!=0])
            ax.set_xticks(xtick)
            ax.set_xticklabels(states[info != 0],rotation=30,fontsize=10)
            ax.grid('on')
            ax.set_ylim(0,400)
            ax.set_title(i2c[e])
        fig.subplots_adjust(hspace=1.2)
        fig.suptitle('EDA')

# shuffle
def shuffle(list):
    new = []
    ind_shf = np.arange(len(list[0]))
    np.random.shuffle(ind_shf)
    for l in list:
        new.append(l[ind_shf])
    return new

##


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

def score_function(pred,real):
    score = f1_score(real, pred, average="macro")
    return score

class Loss1(nn.Module):
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
        return f(modely,targety)

    def forward(self, modely, targety):
        """
        :param modely : B,64 (15+49)
        :param targety: (label_c, label_s)
        :return:
        """
        target_c =  targety[0]
        target_s = targety[1]
        loss_c = self.get_loss(modely[:,:15],target_c, func='entropy')
        loss_s = self.get_loss(modely[:,15:],target_s, func='entropy')

        loss = .5*loss_c+loss_s
        return loss

class Loss2(nn.Module):
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
        return f(modely,targety)

    def forward(self, modely, targety):
        """
        :param modely : B,88
        :param targety: (label_c, label_s)
        :return:
        """
        #target_s = targety[1]
        loss = self.get_loss(modely,targety, func='entropy')
        return loss

class net_(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.pretrain = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=88)
        #self.pretrain = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=88)
        self.init_weights()

    def forward(self,x):
        out_c = self.pretrain(x['img'])
        #
        return out_c

    def loss_f(self, modely, targety):
        f = Loss2()
        return f(modely, targety)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
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
        label_c = x['label_c']
        y_hat = self(x)
        loss = self.loss_f(y_hat, label_c)
        pred_c = y_hat.argmax(-1).detach().cpu().tolist()

        f1_c = score_function(label_c.cpu().tolist(), pred_c)
        return loss, f1_c

    def training_step(self, batch, batch_idx):
        loss,f1_c = self.step(batch)
        self.log('trn_loss', loss, on_step=False, on_epoch=True)
        self.log('trn_f1',   f1_c, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss,f1_c = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_f1',   f1_c, on_step=False, on_epoch=True)
        return {'val_loss': loss,'f1':f1_c}
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([op['val_loss'] for op in outputs]).mean()
        avg_f1 = torch.stack([op['f1'] for op in outputs]).mean()

        print("\n* EPOCH %s | loss :{%4.4f} | f1 :{%2.2f}" % (self.current_epoch, avg_loss, avg_f1))
        return {'loss': avg_loss}

