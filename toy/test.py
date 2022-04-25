import argparse
from torch.utils.data import DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
from sklearn.model_selection import train_test_split
from toy.utils import *
## visualize params
pick = 5
c_threshold = 0.3
save = True
save_dir = r'C:\Users\82109\PycharmProjects\object-detection\toy\sample'

## load config & figure info
ckpdir = r'C:\Users\82109\PycharmProjects\object-detection\toy\log\version_0'
root_dir = r'D:\cv\Dataset/coco_2017/val2017/'
ano_fnm = r'D:\cv\Dataset/coco_2017/annotations/instances_val2017.json'
#
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--pick',       default=pick,           type=int,      help = '')
parser.add_argument('--top_n',      default=0,              type=int,      help = '')
parser.add_argument('--c_threshold',default=c_threshold,    type=float,      help = '')
parser.add_argument('--resize_n',   default=672,            type=int,      help = '')
parser.add_argument('--save',       default=save,           type=int,      help = '')
parser.add_argument('--save_dir',   default=save_dir,       type=str,      help = '')

config = pd.read_csv(f'{ckpdir}/meta_tags.csv')
for i in config.values:
    if i[1]<1:
        parse_type = float
    else:
        parse_type = int
    parser.add_argument('--%s'%i[0],   default=parse_type(i[1]),     type=parse_type,      help = '')
hparams = parser.parse_args(args=[]) # for test
## Test set
seed_everything(seed=hparams.random_seed)

image_list, id2ctg, ctg2name = get_coco_list(ano_fnm, limit=500, repeat=0, num_class = hparams.num_class)
ctg2id = {list(id2ctg.values())[i]:list(id2ctg.keys())[i] for i in range(len(id2ctg))}
id2name = {i:ctg2name[id2ctg[i]] for i in id2ctg.keys()} # 1~80
id2name[0]='background'
train_list_, test_list =  train_test_split(image_list,test_size=0.17012)
#
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
tstset = coco(test_list, model=pretrained, trans=alb, key=key)

tst_loader = DataLoader(tstset,batch_size=hparams.batch_size, shuffle=False,  drop_last=False, num_workers=0, pin_memory=True)
## Load checkpoint
best_pth = glob(r'%s\checkpoints/*.ckpt'%ckpdir)[0]
result = pd.read_csv(r'%s/../../metrics.csv'%best_pth, usecols=['epoch','trn_loss','val_loss'])
c = result.groupby('epoch').mean().plot(marker='*',color=['orange','b'])
c.legend(['trn_loss','val_loss'],fontsize=15)
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('Loss',fontsize=18)
plt.grid('on')
# plt.savefig(r'%s/../loss.jpg'%save_dir,bbox_inches='tight')

#
model = toy2(hparams)
model = model.load_from_checkpoint(best_pth, hparams=hparams)
print(best_pth)
##
visualize(model,
          save=hparams.save,
          save_dir=hparams.save_dir,
          dataset=tstset,
          label_list=test_list,
          key=key,
          id2name=id2name,
          resize_n = hparams.resize_n,
          num_class = hparams.num_class,
          pick = hparams.pick,
          top_n = hparams.top_n,
          c_threshold=hparams.c_threshold,
          root_dir = r'D:\cv\Dataset/coco_2017/val2017/')
