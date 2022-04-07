import numpy as np
import pandas as pd
import os
import gc
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
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
import cv2 as cv
from PIL import Image as pil
from tqdm import tqdm as tqdm
from xai.utils import *
## VOD Image load
seed_everything()

# segment 개념으로 출발
# 각 class마다 따로 segment
# 동일 segment에서, 여러 객체가 있을 경우 2~N, 없을 경우 0
# 겹치는 지역은 1로 따로 표시...?


fdir = r'D:\cv\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'

alb = A.Compose([A.Resize(224,224),
                 A.Normalize(),
                 ToTensorV2()],
                bbox_params=A.BboxParams(format='pascal_voc',
                                         label_fields=['category_ids']),)

trnset = voc_dataset(fdir, phase='train', several=True, trans=alb, imshow=True)
trnset[0].keys()


for i in range(5):
    cv.imshow(str(i),trnset[i]['imgs_raw'])
cv.destroyAllWindows()
##
