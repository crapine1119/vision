import argparse
#
from torch.utils.data import DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
#
from sklearn.model_selection import train_test_split
from toy.utils import *
##
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--num_class',      default=4,     type=int,      help = '')
parser.add_argument('--out_size',       default=64,     type=int,      help = '')
parser.add_argument('--alpha',          default=.75,    type=float,    help = 'ratio of ground truth in 1.0')
parser.add_argument('--gamma',          default=0,       type=float,      help = '')

parser.add_argument('--max_epochs',     default=50,      type=int,      help = '')
parser.add_argument('--batch_size',     default=280,     type=int,      help = '')
#
parser.add_argument('--lr',             default=0.001,   type=float,    help = '')
parser.add_argument('--wd',             default=0.005,   type=float,    help = '')
parser.add_argument('--lr_patience',    default=5,       type=int,    help = '')
parser.add_argument('--es_patience',    default=20,      type=int,    help = '')

parser.add_argument('--random_seed',    default=42,      type=int,      help = '')
# hparams = parser.parse_args()
hparams = parser.parse_args(args=[]) # 테스트용


## Image net tensor Image load
seed_everything(seed=hparams.random_seed)

root_dir = r'D:\cv\Dataset/coco_2017/val2017/'
ano_fnm = r'D:\cv\Dataset/coco_2017/annotations/instances_val2017.json'

image_list, id2ctg, ctg2name = get_coco_list(ano_fnm, limit=500, repeat=0, num_class = hparams.num_class)
ctg2id = {list(id2ctg.values())[i]:list(id2ctg.keys())[i] for i in range(len(id2ctg))}
id2name = {i:ctg2name[id2ctg[i]] for i in id2ctg.keys()} # 1~80
id2name[0]='background'

train_list_, test_list =  train_test_split(image_list,test_size=0.17012)
train_list, valid_list = train_test_split(train_list_,test_size=0.2)
##
alb = A.Compose([A.Resize(224,224),
                 A.Normalize(),
                 ToTensorV2()],
                bbox_params=A.BboxParams(format='coco',
                                         label_fields=['category_ids']),)

pretrained = models.resnet34(pretrained=True)
pretrained.eval()
for i in pretrained.parameters():
    i.requires_grad_(False)

key = 'layer2,layer3,layer4'.split(',')
trnset = coco(train_list, model=pretrained, trans=alb, key=key)
valset = coco(valid_list, model=pretrained, trans=alb, key=key)
# tstset = coco(test_list, model=pretrained, trans=alb, key=key)
trn_loader = DataLoader(trnset,batch_size=hparams.batch_size, shuffle=True,  drop_last=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(valset,batch_size=hparams.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
##
sdir = r'D:\cv\toy\naverlabs'
name     = 'log'
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

model = toy2(hparams)
trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

best_pth = checkpoint_callback.kth_best_model_path