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
    def __init__(self,fdir = r'D:\cv\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007', phase='train', trans=None ,imshow=False):
        super().__init__()
        self.fdir = fdir
        self.trn_files = pd.read_csv(f'{fdir}\ImageSets\Main/train.txt', header=None, dtype=object).values[:, 0]
        self.val_files = pd.read_csv(f'{fdir}\ImageSets\Main/val.txt', header=None, dtype=object).values[:, 0]
        self.names = [os.path.split(i)[-1][:-13] for i in glob(f'{fdir}\ImageSets\Main/*_trainval*')]
        self.labels = {i:self.names[i] for i in range(len(self.names))}
        self.n2l = {self.names[i]:i for i in range(len(self.names))} # name to label
        self.phase = phase
        self.trans = trans
        self.imshow = imshow
    def __len__(self):
        if self.phase=='train': return self.trn_files.shape[0]
        else : return self.val_files.shape[0]
    #
    def __getitem__(self, item):
        if self.phase=='train':
            img_fnm = self.trn_files[item]
        else:
            img_fnm = self.val_files[item]
        img_dir = f'{self.fdir}/JPEGImages/{img_fnm}.jpg'
        img_raw = cv.imread(img_dir,cv.IMREAD_COLOR)
        img_raw = cv.cvtColor(img_raw,cv.COLOR_BGR2RGB)
        if self.trans:
            img = self.trans(pil.fromarray(img_raw))
        #
        tree = parse(f'{self.fdir}/Annotations/{img_fnm}.xml')
        root = tree.getroot()
        #
        labels,bboxes = [],[]
        for ob in root.findall('object'):
            xml_bbox = ob.find('bndbox')
            bbox = []
            for xy in ['xmin','ymin','xmax','ymax']:
                bbox.append(int(xml_bbox.find(xy).text))
            if self.imshow:
                img_raw = cv.rectangle(img_raw,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=[0,255,0],thickness=2)
                img_raw = cv.putText(img_raw, ob.find('name').text, (bbox[0], bbox[1]-10),
                                     fontFace=cv.FONT_ITALIC,
                                     fontScale=.5,
                                     color=[0,255,0],
                                     thickness=2)
            bboxes.append(bbox)
            labels.append(self.n2l[ob.find('name').text])
        info = {'imgs': img, 'labels': torch.LongTensor(labels)[0], 'bboxes': torch.FloatTensor(bboxes)[0]} # just one obj classification
        if self.imshow:
            info['imgs_raw'] = img_raw
        return info
        #

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
class vgg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = models.vgg11(pretrained=True)
        self.net = model.features[:20]
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                squeeze(),
                                nn.Linear(512, num_classes, bias=True),
                                nn.Softmax())

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x.squeeze()
#
class vgg_grad(nn.Module):
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

## cam methods
class cam():
    def __init__(self, model, img, label):
        model.cpu()
        model.eval()

        # output of last conv (features) : A
        features = model.net(img).detach() # 64,512,14,14

        # weight of fc (class, last channel) : w(c)
        # can also be done like 'model.state_dict()['fc.2.weight']
        fc_params = {i[0]:i[1] for i in model.fc[2].named_parameters()} # model의 fc의 2번째 layer w를 가져옴
        class_weight = fc_params['weight'].detach() # 10,512

        # multiply feature map & weight of each class
        feature_mul = (features*
                       class_weight[label].view(len(label),512,1,1)).sum(1)

        # upsample feature_mul to train image
        upsample = nn.Upsample(scale_factor=img.shape[-1]/features.size()[-1], mode='bilinear')

        # cam result of all classes
        cam_result = upsample(feature_mul.view(len(label),1,14,14))

        # unnormalize img
        self.unnorm = (img.permute(0,2,3,1)*.5 +.5).numpy()

        cam_min,_ = cam_result.view(len(label),-1).min(dim=-1)
        cam_max,_ = cam_result.view(len(label),-1).max(dim=-1)

        # unnormed cam
        self.final = (cam_result-cam_min.view(len(label),1,1,1))/\
                     (cam_max-cam_min).view(len(label),1,1,1)

    def show(self,item):
        plt.figure(figsize=[9, 9])
        plt.imshow(self.unnorm[item], alpha=1.0)
        c = plt.contourf(self.final[item, 0].numpy(), cmap='jet', levels=np.arange(.5, 1+.001, .025), alpha=.6)
        c.cmap.set_under('b')
        plt.colorbar(c)
#
class gradcam():
    def __init__(self, model_grad, img, label):
        model_grad.cpu()
        model_grad.eval()
        print('Get grad...')
        output = model_grad(img)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, label)
        loss.backward()

        A = model_grad.activations[0] # 64,512,7,7
        dY_dA = model_grad.gradients[0][0]  # 64,512,7,7

        feature_mul = nn.ReLU()(A*dY_dA).mean(dim=1) # 64,7,7

        # upsample feature_mul to train image
        upsample = nn.Upsample(scale_factor=img.shape[-1]/A.size()[-1], mode='bilinear')

        # cam result of all classes
        cam_result = upsample(feature_mul.view(len(label),1,7,7)).detach()

        # unnormalize img
        self.unnorm = (img.permute(0,2,3,1)*.5 +.5).numpy()

        cam_min,_ = cam_result.view(len(label),-1).min(dim=-1)
        cam_max,_ = cam_result.view(len(label),-1).max(dim=-1)

        # unnormed cam
        self.final = (cam_result-cam_min.view(len(label),1,1,1))/\
                     (cam_max-cam_min).view(len(label),1,1,1)
    def show(self,item):
        plt.figure(figsize=[9, 9])
        plt.imshow(self.unnorm[item], alpha=1.0)
        c = plt.contourf(self.final[item, 0].numpy(), cmap='jet', levels=np.arange(.5, 1+.001, .025), alpha=.6)
        c.cmap.set_under('b')
        plt.colorbar(c)