import argparse
#
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
#
from sklearn.model_selection import train_test_split
from yolo.yolo_v2.utils import *
##
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--out_size',       default=128,     type=int,      help = '')

parser.add_argument('--max_epochs',     default=100,      type=int,      help = '')
parser.add_argument('--batch_size',     default=160,     type=int,      help = '')
#
parser.add_argument('--lr',             default=0.01,   type=float,    help = '')
parser.add_argument('--wd',             default=0.005,   type=float,    help = '')
parser.add_argument('--lr_patience',    default=5,       type=int,    help = '')
parser.add_argument('--es_patience',    default=10,      type=int,    help = '')

parser.add_argument('--random_seed',    default=42,      type=int,      help = '')
# hparams = parser.parse_args()
hparams = parser.parse_args(args=[]) # 테스트용


## Image net tensor Image load
seed_everything(seed=hparams.random_seed)

root_dir = r'D:\cv\Dataset/coco_2017/val2017/'
ano_fnm = r'D:\cv\Dataset/coco_2017/annotations/instances_val2017.json'

# image_list, id2ctg, ctg2name = get_items(ano_fnm, limit=1500, repeat=5)
image_list, id2ctg, ctg2name = get_coco_list(ano_fnm, limit=200, repeat=0)
ctg2id = {list(id2ctg.values())[i]:list(id2ctg.keys())[i] for i in range(len(id2ctg))}
id2name = {i:ctg2name[id2ctg[i]] for i in id2ctg.keys()} # 1~80
id2name[0]='background'

train_list_, test_list =  train_test_split(image_list,test_size=0.15164)
train_list, valid_list = train_test_split(train_list_,test_size=0.2)
##
alb = A.Compose([A.Resize(416,416),
                 A.Normalize(),
                 ToTensorV2()],
                bbox_params=A.BboxParams(format='coco',
                                         label_fields=['category_ids']),)

pretrained_ = models.resnet18(pretrained=True)
pretrained = nn.Sequential(*[*pretrained_.children()][:-2])
for i in pretrained.parameters():
    i.requires_grad_(False)


trnset = coco(train_list, image_size=416, model=pretrained, trans=alb)
valset = coco(valid_list, image_size=416, model=pretrained, trans=alb)
tstset = coco(test_list, image_size=416, model=pretrained, trans=alb)

trn_loader = DataLoader(trnset,batch_size=hparams.batch_size, shuffle=True,  drop_last=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(valset,batch_size=hparams.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)