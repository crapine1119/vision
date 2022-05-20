import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image as pil
from glob import glob as glob
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split
from torchvision import models
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, accuracy_score
import timm
import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings('ignore')
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

## raw
# fdir = r'D:\cv\Dataset/손흥민 챔스 모든골 14시즌부터 전부 모아옴 (초고화질).mp4'
# cv_name = 'son'
# cap = cv.VideoCapture(fdir)
# #fourcc = cv.VideoWriter_fourcc(*'DIVX')
#
# fps = int(cap.get(cv.CAP_PROP_FPS)*1.5) # 1.5
# if cap.isOpened():
#     while True:
#         ret,frame = cap.read()
#         fps_prop = cap.get(cv.CAP_PROP_POS_FRAMES)
#         print(fps_prop)
#         if fps_prop%fps==0:
#             cv.imshow(cv_name,frame)
#         if cv.waitKey(1)==27:
#             break
#     cap.release()
#     cv.destroyAllWindows()
##
seed_everything()

fdir = r'D:\cv\Dataset\SON'

dset = glob('%s/*.png'%fdir)

info = [os.path.split(i)[-1][:-4] for i in dset]

goal = [i[4] for i in info]
zone = [i[6] for i in info]
foot = [i[8] for i in info]

Dset = pd.DataFrame([dset,goal,zone,foot]).T

Dset.index = Dset[0]
Dset.drop(columns=0,inplace=True)
Dset.columns = ['goal','zone','foot']
Dset = Dset.astype(int)

print(Dset[Dset.goal==1].shape)
print(Dset[Dset.goal==0].shape)
## EDA

# Dset[Dset.goal==1].zone.hist(bins=np.arange(-.25,7,.5))
# Dset[Dset.goal==1].foot.hist(bins=np.arange(-.25,2,.5))

trnx_,tstx,trny_,tsty = train_test_split(Dset.index,Dset.goal,test_size=0.18)
##
class custom(Dataset):
    def __init__(self,Dset_x,Dset_y,trans,train=True):
        super().__init__()
        self.files = Dset_x
        self.labels = Dset_y.values
        self.trans = trans
        self.trans_test = A.Compose([A.Resize(540,960,interpolation=cv.INTER_AREA),
                                     A.Normalize(),
                                     ToTensorV2()])
        self.train = train
    def __len__(self):
        return len(self.files)
    def __getitem__(self, item):
        img = cv.imread(self.files[item],cv.IMREAD_COLOR)
        h, w, _ = img.shape
        if h>=720 and w>=1280:
            s_h = int(h / 2 - 360)
            e_h = s_h+720
            s_w = int(w / 2 - 640)
            e_w = s_w+1280
            img = img[s_h:e_h,s_w:e_w]
        else:
            img = cv.resize(img, (720,1280))
        if self.train:
            img = self.trans(image=img)['image']
        else:
            img = self.trans_test(image=img)['image']
        label = torch.tensor([self.labels[item]],dtype=torch.long)[0]
        return {'img':img,
                'label':label}

def score_function(pred,real):
    score = f1_score(real, pred, average="macro")
    return score

def get_callbacks(path=r'C:\Users\82109\PycharmProjects\object-detection\todo\EA',name='log'):
    log_path = '%s'%(path)
    tube     = Tube(name=name, save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='{epoch:02d}-{val_loss:.4f}',
                                          save_top_k=1,
                                          mode='min')
    early_stopping        = EarlyStopping(monitor='val_loss',
                                          patience=15,
                                          verbose=True,
                                          mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    print('get (1)ckp, (2)es, (3)lr_monitor callbacks with (4)tube')
    return {'callbacks':[checkpoint_callback, early_stopping, lr_monitor],
            'tube':tube}

class net(LightningModule):
    def __init__(self):
        super().__init__()
        #self.pretrain = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=2)
        self.pretrain = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=88, drop_rate=0.8)
        self.result = []
    def forward(self,x):
        out_c = self.pretrain(x['img'])
        return out_c

    def loss_f(self, modely, targety):
        f = nn.CrossEntropyLoss()
        return f(modely, targety)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-1)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=4,
                                                   min_lr=1e-7)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        label = x['label'].squeeze()
        y_hat = self(x)
        loss = self.loss_f(y_hat, label)
        pred_c = y_hat.argmax(-1).detach().cpu().tolist()
        f1_c = score_function(label.cpu().tolist(), pred_c)
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

class hook(nn.Module):
    def __init__(self, model, target_layer):
        """
        :param model: pretrained
        :param target_layer: ex) model.features[-1]
        """
        super().__init__()
        self.net = model
        self.target_layer = target_layer
        self.activations = []
        self.gradients = []
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)
        #self.net.classifier[c_num].register_forward_hook(self.forward_hook)
        #self.net.classifier[c_num].register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output): # get activation
        self.activations.append(output)

    def backward_hook(self, module, grad_input, grad_output): # In here, assume using specified module
        self.gradients.append(grad_output)

    def forward(self, x):
        x = self.net(x)
        return x.squeeze()

def load_ckp(path=r'C:\Users\82109\PycharmProjects\object-detection\todo\EA\log',fig=True,n=1):
    if fig:
        fig = plt.figure(figsize=[18,6])
        ax1_leg,ax2_leg = [],[]
        for i in range(n,0,-1):
            best_pth = glob(r'%s\*/*/*.ckpt'%path)[-i]
            version = os.path.dirname(best_pth)[-13]
            result = pd.read_csv(r'%s/../../metrics.csv'%best_pth)
            ax1 = fig.add_subplot(121);ax2 = fig.add_subplot(122)
            #
            result.groupby('epoch').mean().plot(y=['trn_loss','val_loss'],ax=ax1,marker='*')
            ax1_leg.extend(['trn_loss(%s)' % version, 'val_loss(%s)' % version])
            result.groupby('epoch').mean().plot(y=['trn_f1','val_f1'],ax=ax2,marker='*')
            ax2_leg.extend(['trn_f1(%s)' % version, 'val_f1(%s)' % version])
        ax1.grid('on')
        ax1.axhline(0,color='r')
        ax2.grid('on')
        ax2.axhline(1,color='r')
        ax1.legend(ax1_leg)
        ax2.legend(ax2_leg)
    else:
        best_pth = glob(r'%s\*/*/*.ckpt'%path)[-1]
    return best_pth

##
trnx,trny = [],pd.DataFrame([])
for i in range(2):
    trnx.extend(trnx_)
    trny = pd.concat([trny,trny_])


##
trans = A.Compose([A.CenterCrop(700,800,p=0.5),
                   #A.RandomCrop(700,800, p=0.5),
                   A.HorizontalFlip(p=0.5),
                   A.ShiftScaleRotate(shift_limit=0.04,
                                      scale_limit=0.1,
                                      rotate_limit=10,
                                      interpolation=cv.INTER_AREA,
                                      border_mode=cv.BORDER_CONSTANT,
                                      value=0,
                                      p=1.0),
                   A.ChannelShuffle(p=0.5),
                   A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                   #A.ToGray(p=0.2),
                   A.Resize(540,960,interpolation=cv.INTER_AREA),
                   A.Normalize(),
                   ToTensorV2()])

trnset = custom(trnx,trny, trans=trans, train=True)
tstset = custom(tstx,tsty, trans=trans, train=True)

n = np.random.randint(68)
fig = plt.figure(figsize=[18,8])
ax1 = fig.add_subplot(221)
ax1.imshow(trnset[n]['img'].permute(1,2,0).numpy()*0.22+0.45)
ax2 = fig.add_subplot(222)
ax2.imshow(trnset[n+68]['img'].permute(1,2,0).numpy()*0.22+0.45)
ax3 = fig.add_subplot(223)
ax3.imshow(trnset[min(n+68*2,len(trny)-1)]['img'].permute(1,2,0).numpy()*0.22+0.45)
ax4 = fig.add_subplot(224)
ax4.imshow(trnset[min(n+68*3,len(trny)-1)]['img'].permute(1,2,0).numpy()*0.22+0.45)
##

trn_loader = DataLoader(trnset,batch_size=8,shuffle=True,drop_last=True,num_workers=0,pin_memory=True)
val_loader = DataLoader(tstset,batch_size=8,shuffle=False,drop_last=True,num_workers=0,pin_memory=True)

callbacks = get_callbacks()

print('Call trainer...')
trainer = Trainer(max_epochs=50,
                  callbacks=callbacks['callbacks'],
                  gpus=2,
                  precision=16,
                  logger=callbacks['tube'],
                  deterministic=True, accelerator='dp', accumulate_grad_batches=2)
print('Train model...')

model = net()
trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

checkpoint_callback = callbacks['callbacks'][0]
best_pth = checkpoint_callback.kth_best_model_path
##

best_pth = load_ckp(fig=False,n=2)
##
tstset = custom(tstx,tsty, trans=trans, train=False)
tst_loader = DataLoader(tstset,batch_size=16,shuffle=False,drop_last=False,num_workers=0,pin_memory=True)
model = net()
model = model.load_from_checkpoint(best_pth)
model.eval()
if model.result.__len__()>0:
    model.result = []

callbacks = get_callbacks()
trainer = Trainer(max_epochs=1,
                  callbacks=callbacks['callbacks'],
                  gpus=1,
                  precision=16,
                  logger=callbacks['tube'],
                  deterministic=True, accelerator='dp', accumulate_grad_batches=2)
trainer.test(model=model,
             dataloaders=tst_loader,
             ckpt_path=best_pth)

pred = model.result
score_function(pred,tsty.values.tolist())
##

trnset = custom(trnx,trny, trans=trans, train=False)
trn_loader = DataLoader(trnset,batch_size=4,shuffle=False,drop_last=False,num_workers=0,pin_memory=True)

class gradcam():
    def __init__(self, model_grad, img, label):
        self.label = label
        model_grad.cuda()
        model_grad.eval()
        print('Get grad...')
        output = model_grad(img)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, label)
        loss.backward()

        A = model_grad.activations[0] # 64,512,7,7
        _,_,h,w = A.size()
        dY_dA = model_grad.gradients[0][0]  # 64,512,7,7

        feature_mul = nn.ReLU()(A*dY_dA).mean(dim=1) # 64,7,7

        # upsample feature_mul to train image
        upsample = nn.Upsample(scale_factor=img['img'].shape[-1]/w, mode='bilinear')

        # cam result of all classes
        cam_result = upsample(feature_mul.view(len(label),1,h,w)).detach()

        # unnormalize img
        self.unnorm = (img['img'].permute(0,2,3,1)*.5 +.5).numpy()

        cam_min,_ = cam_result.view(len(label),-1).min(dim=-1)
        cam_max,_ = cam_result.view(len(label),-1).max(dim=-1)

        # unnormed cam
        self.final = (cam_result-cam_min.view(len(label),1,1,1))/\
                     (cam_max-cam_min).view(len(label),1,1,1)
    def show(self,item):
        plt.figure(figsize=[9, 9])
        plt.imshow(self.unnorm[item], alpha=1.0)
        c = plt.contourf(self.final[item, 0].numpy(), cmap='jet', levels=np.arange(.75, 1+.001, .025), alpha=.6)
        c.cmap.set_under('b')
        plt.colorbar(c)
        plt.title(self.label[item])

target_layer = model.pretrain.conv_head
model_grad = hook(model, target_layer)


for img in tst_loader:break
for img_train in trn_loader:break

for i in img:
    img[i] = img[i].cuda()
    img_train[i] = img_train[i].cuda()


result_grad = gradcam(model_grad,img,img['label'])
result_grad_trn = gradcam(model_grad,img_train,img_train['label'].squeeze())


for i in [i for i in range(16) if pred[i]==tsty[i]]:
    print(tsty[i])
    result_grad.show(i)


