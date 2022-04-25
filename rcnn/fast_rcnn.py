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
seed_everything(seed=42)

root_dir = r'D:\cv\Dataset/coco_2017/val2017/'
ano_fnm = r'D:\cv\Dataset/coco_2017/annotations/instances_val2017.json'

def get_items(ano_fnm,get=3):
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
        if labels==get:
            image_list.append({'image_id':image_id, 'bbox':bbox, 'labels':labels})
    return np.asarray(image_list), id2ctg, ctg2name

image_list, id2ctg, ctg2name = get_items(ano_fnm)

ctg2id = {list(id2ctg.values())[i]:list(id2ctg.keys())[i] for i in range(len(id2ctg))}

trn_x, tst_x =  train_test_split(image_list,test_size=0.4)
vld_x = tst_x[:len(tst_x)//2]


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

##
t1 = trn_x[0]
print(ctg2name[id2ctg[t1['labels'][0]]])
# annotations에는 index 정보만 줌 : 이름을 확인하려면 index를 카테고리로 바꾸면 됨

fnm = '%s/%012d.jpg'%(root_dir,t1['image_id'])
img_rgb = cv.imread(fnm)

# ground truth
grt = t1['bbox'].tolist()
grt[2] += grt[0]
grt[3] += grt[1]


r1,r2,r3,r4 = map(int,grt)

red = (255, 0, 0)
img_rgb = cv.rectangle(img_rgb, (r1, r2), (r3, r4), color=red, thickness=2)
cv.imshow('a',img_rgb)
##
min_num = 8
ref = 224

def do_ss(min_num,dict):
    global ref
    fnm = '%s/%012d.jpg'%(root_dir,dict['image_id'])
    img = cv.imread(fnm)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #_ , regions = ss(img_rgb, scale=100, min_size=2000)
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_rgb)
    ss.switchToSelectiveSearchFast() # 빠른 대신 재현율은 낮음
    rects = ss.process() # x,y,w,h

    grt = np.array(dict['bbox'])
    grt[2] += grt[0]
    grt[3] += grt[1]
    #grt = grt.astype('int').tolist() # 그림 그릴때만 필요
    #
    count_t,count_f = 0,0,
    flag,fflag,bflag = 0,0,0
    trn_imgs,trn_rects, trn_labels = [],[], []
    for e,box in enumerate(rects):
        if flag==0: # 2000개 까지만으로 제한
            roi = list(box)
            roi[2] += roi[0]
            roi[3] += roi[1]
            iou = get_iou(roi,grt)

            if count_t < min_num:
                if iou>=0.5:
                    #cut = img_rgb[roi[1]:roi[3],roi[0]:roi[2]]
                    #resized = cv.resize(cut, (224,224), interpolation= cv.INTER_AREA)
                    #trn_imgs.append(resized)
                    trn_rects.append(roi)
                    trn_labels.append(dict['labels'].item())
                    count_t +=1
            else:
                fflag = 1
            #
            if count_f < min_num:
                if iou<0.1:
                    # cut = img_rgb[roi[1]:roi[3], roi[0]:roi[2]]
                    # resized = cv.resize(cut, (224, 224), interpolation=cv.INTER_AREA)
                    # trn_imgs.append(resized)
                    trn_rects.append(roi)
                    trn_labels.append(0)
                    count_f+=1
            else:
                bflag = 1
        if fflag==1 and bflag==1:
            break
    if fflag==0 or bflag==0:
        return [],[],[],[]
    return np.array(img_rgb),np.array(trn_rects, dtype=np.int_), np.array(grt), np.array(trn_labels, dtype=np.int_)
# tx,ty = do_ss(t1)
# text = "%s: %.2f"%(e, iou)
# cv.putText(img_rgb, text, (roi[0] + 100, roi[1] + 10),
#            fontFace = cv.FONT_HERSHEY_SIMPLEX,
#            fontScale= 0.4,
#            color = [0,255,0],
#            thickness = 1)



def sep_pn(min_num,when,flst,sdir,make_sample=False):
    if os.path.isdir(sdir)==False:
        os.makedirs(sdir)
    count=0
    for dict in tqdm(flst):
        if count == when:
            break
        img_rgb, rects, grt, labels = do_ss(min_num, dict) # 32개 나올때 까지

        if len(labels)>0:
            if (labels != 0).sum() >= min_num:
                print(dict['image_id'], True)
                count += 1
                if make_sample:
                    #fnm = '%s/%012d.jpg' % (root_dir, dict['image_id'])
                    # img_rgb = cv.imread(fnm)
                    # img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
                    cv.imshow(str(dict['image_id']), img_rgb)
                    cv.imwrite(r'%s/%012d.jpg' % (sdir, dict['image_id']), img_rgb)
                    cv.destroyAllWindows()
                idx = id2ctg[labels[labels!=0][0]]
                pos = pd.DataFrame(rects[labels != 0])
                neg = pd.DataFrame(rects[labels == 0])
                grt = pd.DataFrame([grt])
                pos.to_csv('%s/%012d_%s_p.csv'%(sdir,dict['image_id'],idx),index=False)
                neg.to_csv('%s/%012d_%s_n.csv'%(sdir,dict['image_id'],idx),index=False)
                grt.to_csv('%s/%012d_grt.csv'%(sdir,dict['image_id']),index=False)
sdir = r'D:\cv\01rcnn\sample'

sep_pn(4,50,trn_x,sdir,make_sample=True)
sep_pn(8,20,vld_x,r'D:\cv\01rcnn\sample_valid',make_sample=True)
sep_pn(8,20,tst_x,r'D:\cv\01rcnn\sample_test',make_sample=True)


## customizing
import os
ex_fnm = glob.glob(r'%s/*p.csv'%(sdir))[10]
ctg2name[int(os.path.split(ex_fnm)[-1][13:-6])]
img_id = os.path.split(ex_fnm)[-1][:12]

ex_img = cv.imread('%s/%s.jpg'%(sdir,img_id))
x1,y1,x2,y2 = pd.read_csv('%s/%s_grt.csv'%(sdir,img_id)).values[0].astype(int)

ex_img = cv.rectangle(ex_img,(x1,y1),(x2,y2),color=[0,0,255],thickness=2)
roi = pd.read_csv(ex_fnm)
for r in roi.values:
    ex_img = cv.rectangle(ex_img,(r[0],r[1]),(r[2],r[3]),color=[0,255,0],thickness=1)
cv.imshow('ROI',ex_img)
##

def resize(fnm, ref):
    if isinstance(fnm,str):
        tmp = cv.imread(fnm)
    else: # numpy
        tmp = fnm.copy()
    if tmp.shape[0] > ref or tmp.shape[1] > ref:
        fx = min(0.05 + ref / tmp.shape[0], 1)
        fy = min(0.05 + ref / tmp.shape[1], 1)
        ff = max(fx, fy)
        tmp = cv.resize(tmp, dsize=(0, 0), fx=ff, fy=ff, interpolation=cv.INTER_LINEAR)
    diff_w = abs(tmp.shape[0] - ref)
    diff_h = abs(tmp.shape[1] - ref)
    if tmp.shape[0] > ref:
        tmp = tmp[diff_w // 2:diff_w // 2 + ref, :]
    else:
        w1 = diff_w // 2
        w2 = diff_w - w1
        tmp = cv.copyMakeBorder(tmp, 0, 0, w1, w2, cv.BORDER_CONSTANT, value=[0, 0, 0])

    if tmp.shape[1] > ref:
        tmp = tmp[:, diff_h // 2:diff_h // 2 + ref]
    else:
        h1 = diff_h // 2
        h2 = diff_h - h1
        tmp = cv.copyMakeBorder(tmp, h1, h2, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
    return tmp
class customdataset(Dataset):
    def __init__(self,sdir,transform=None):
        self.samples = glob.glob('%s/*.jpg'%sdir)

        # load all (car) images
        # self.jpeg_images = [torch.tensor(cv.imread(sample_name),dtype=torch.float32) for sample_name in self.samples]
        # positive : iou >= 0.5 negative : iou < 0.5
        # Save positive and negative separately
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
        img_raw = resize(tmp,224)
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
        onehot[:len(self.p_img)] = ctg2id[self.labels[i]]

        return {'img': imgs,
                'img_raw': img_raw,
                'label': onehot,
                'grt': self.grt[i],
                'p_rect':self.positive_rects[i],
                'p_size':self.positive_sizes[i],
                'n_rect':self.negative_rects[i],
                'n_size':self.negative_sizes[i]}

a = customdataset(sdir)
b = customdataset(sdir+'_valid')
c = customdataset(sdir+'_test')
# ctg2name[a[3]['label']]

trn_loader = DataLoader(a, batch_size=1, shuffle=True, drop_last= True)
vld_loader = DataLoader(b, batch_size=1, shuffle=True, drop_last= True)
tst_loader = DataLoader(c, batch_size=c.__len__(), shuffle=False, drop_last= False)
##
test = a[0]['img_raw']
test.shape
roi = a[0]['p_rect'][0]

test[:,roi[1]:roi[3],roi[0]:roi[2]]

roi


test_ss =

model = models.vgg16_bn(pretrained=True)

get_featuremap = model.features[:-1]

test_features = get_featuremap(test)

test_features.shape # 16, 512 , 14, 14


class slowroipool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveAvgPool2d(output_size)
        self.size = output_size

    def foward(self,img,roi_ary,roi_idx):
        # roi_ary [n,x1,y1,x2,y2]
        n = roi_ary.shape[0]
        h,w = img.size(2),img.size(3)
        x1 = roi_ary[:, 0]
        y1 = roi_ary[:, 1]
        x2 = roi_ary[:, 2]
        y2 = roi_ary[:, 3]

        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)

        res = []
        for i in range(n):
            one = img[roi_idx[i],:,y1[i]:y2[i],x1[i]:x2[i]].unsqueeze(0)
            one = self.maxpool(one)
            res.append(one)
        res = torch.cat(res,dim=0)
        return res

class fast_rcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = model.features[:-1]
        self.slowroipool = slowroipool(output_size=(7,7))
        self.feature = model.classifier[:-1]







##

model = models.vgg16(pretrained=True)
num_features = model.classifier[6].in_features
#model.classifier[6] = nn.Linear(num_features,len(id2ctg)+1) # 15(N) * 81(C)
model.classifier[6] = nn.Linear(num_features,2) # 15(N) * 81(C)
model = model.cuda() # 반드시 가장 나중에

real_batch = a.positive_sizes[0]+a.negative_sizes[0]
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)

max_epoch = 10
def cl(epoch,trn_loader,vld_loader):
    batch_iter = trn_loader.__len__()
    batch_iter_v = vld_loader.__len__()
    loss = {'train':0.0,'valid':0.0}
    acc = {'train': 0, 'valid': 0}
    wrap = {'train': tqdm(trn_loader),
            'valid': tqdm(vld_loader)}
    for phase in ['train','valid']:
        if phase=='train':
            model.train()
        else:
            model.eval()
        idx_v = 0
        for idx, dict in enumerate(wrap[phase]):
            if phase=='valid':
                idx_v = int(idx)
            input = dict['img'][0].cuda()
            label = dict['label'][0].cuda()
            optimizer.zero_grad()
            #
            with torch.set_grad_enabled(phase == 'train'):
                output = model(input)
                batch_loss = criterion(output, label) # 16개 평균 loss
                batch_acc = torch.sum(torch.argmax(output, dim=1) == label.data) # 16개
                if phase=='train':
                    batch_loss.backward()
                    optimizer.step()
            loss[phase] += batch_loss*label.size(0)
            acc[phase]  += batch_acc
            wrap[phase].set_postfix({
                'Epoch': epoch + 1,
                'Mean Loss': '{:06f}'.format(loss['train'] / (idx + 1)),
                'Mean Acc.': '{:04f}'.format(100*acc['train'] / ((idx + 1) * real_batch)),
                'Mean Val Loss': '{:06f}'.format(loss['valid'] / (idx_v + 1)),
                'Mean Val Acc.': '{:04f}'.format(100*acc['valid'] / ((idx_v + 1) * real_batch))})
    return  loss['train']/batch_iter, acc['train']/(batch_iter*real_batch), loss['valid']/batch_iter_v, acc['valid']/(batch_iter_v*real_batch)

total_loss, total_loss_v = [], []
total_acc, total_acc_v = [], []
for epoch in range(max_epoch):
    loss, acc, loss_v, acc_v = cl(epoch,trn_loader,vld_loader)
    #
    total_loss.append(loss)
    total_loss_v.append(loss_v)
    total_acc.append(acc)
    total_acc_v.append(acc_v)
plt.figure()
plt.plot(total_loss);plt.plot(total_loss_v)
plt.figure()
plt.plot(total_acc);plt.plot(total_acc_v)
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

in_features = 256 * 6 * 6
out_features = 4
bbox_regressor = nn.Linear(in_features, out_features)
bbox_regressor.cuda()
pretrained = model.features
def rg(max_epoch,pretrained,bbox_regressor,trn_loader):
    batch_iter = trn_loader.__len__()
    pretrained.eval()
    for param in pretrained.parameters():
        param.requires_grad = False
    #
    bbox_regressor.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(bbox_regressor.parameters(), lr=1e-5, weight_decay=1e-4)
    #
    total_loss = []
    for epoch in range(max_epoch):
        loss = 0.0
        for idx, dict in enumerate(trn_loader):
            input = dict['img'][0].cuda()
            label = dict['label'][0].cuda()
            p_rect = dict['p_rect'][0].cuda()
            grt = dict['grt'][0].cuda()
            target = calcul(p_rect,grt)
            #
            features = pretrained(input[label!=0]) # positive feature만
            features = torch.flatten(features, 1)
            # zero the parameter gradients
            optimizer.zero_grad()
            output = bbox_regressor(features)
            batch_loss = criterion(output,target)
            batch_loss.backward()
            optimizer.step()
            loss+=batch_loss*target.size(0)
        total_loss.append(loss/(batch_iter))
        print('\t Epoch : %02d, Loss : %.4f'%(epoch,(loss/(batch_iter)).item()))
    return total_loss
    # regressor

box_loss = rg(35,pretrained,bbox_regressor,trn_loader)

## 테스트 결과 확인

from sklearn.metrics import f1_score
def acc(pred,real):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

model.eval()
test = next(iter(tst_loader))
img = test['img'].view(-1,3,224,224).cuda()
label = test['label'].view(-1).cuda()

with torch.no_grad():
    output = model(img)
acc(output,label)
new_crosstab = pd.crosstab(label.cpu(),torch.argmax(output,1).cpu(), rownames=['answer'], colnames=['preds'])
##
pretrained.eval()
bbox_regressor.eval()
with torch.no_grad():
    features = pretrained(img)
    features = torch.flatten(features, 1)
    tset = bbox_regressor(features)

px = (test['p_rect'][:,:,2] + test['p_rect'][:,:,0])/2
py = (test['p_rect'][:,:,3] + test['p_rect'][:,:,1])/2
pw = (test['p_rect'][:,:,2] - test['p_rect'][:,:,0])
ph = (test['p_rect'][:,:,3] - test['p_rect'][:,:,1])

tset_raw = tset.view(c.__len__(), 16, 4).cpu()[:,:8] # 변환장치

tx = tset_raw[:,:,0]
ty = tset_raw[:,:,1]
tw = tset_raw[:,:,2]
th = tset_raw[:,:,3]

gx = (pw*tx + px).unsqueeze(2) # 중앙x
gy = (ph*ty + py).unsqueeze(2) # 중앙y
gw = (pw*torch.exp(tw)).unsqueeze(2) # 전체 w
gh = (ph*torch.exp(th)).unsqueeze(2) # 전체 h

test_g_rect = torch.cat([gx-gw/2,gy-gh/2,gx+gw/2,gy+gh/2],dim=2)

##
for target in range(10):
    target_fnm = glob.glob(r'D:\cv\01rcnn\sample_test/*.jpg')[target]

    x1,y1,x2,y2 = map(int,test['grt'][target].numpy())
    img_rgb = cv.imread(target_fnm)
    img_rgb = cv.rectangle(img_rgb,(x1,y1),(x2,y2),color=[0,0,255],thickness=2)

    #for B,A in zip(test['p_rect'][target],test_g_rect[target]):
        ##
        target = 24
        for tq in range(8):
            target_fnm = glob.glob(r'D:\cv\01rcnn\sample_test/*.jpg')[target]
            x1, y1, x2, y2 = map(int, test['grt'][target].numpy())
            img_rgb = cv.imread(target_fnm)
            img_rgb = cv.rectangle(img_rgb, (x1, y1), (x2, y2), color=[0, 0, 255], thickness=2)
            B,A = test['p_rect'][target][tq], test_g_rect[target][tq]

            b1,b2,b3,b4 = B.numpy()
            a1,a2,a3,a4 = map(int,A.numpy())

            img_rgb = cv.rectangle(img_rgb, (b1, b2), (b3, b4), color=[0, 255, 0], thickness=1)
            img_rgb = cv.rectangle(img_rgb, (a1, a2), (a3, a4), color=[255, 0, 0], thickness=1)
            cv.imshow(str(target)+str(tq),img_rgb)
##



def nms(rects,scores):
    """
    :param rects: (8,4)
    :param scores: (8)
    :return:
    """
    nms_rects, nms_scores = [],[]

    rects = np.array(rects)
    scores = np.array(scores)

    idxs = np.argsort(scores)[::-1]
    rect_array = rects[idxs]
    score_array = scores[idxs]
    while len(score_array)>0:
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        # pop
        rect_array = rect_array[1:]
        score_array = score_array[1:]
    return nms_rects[0], nms_scores[0]

trans = True

output_raw = output.view(c.__len__(), 16, 2).cpu()[:,:8]
for target in range(10):
    target_fnm = glob.glob(r'D:\cv\01rcnn\sample_test/*.jpg')[target]

    x1,y1,x2,y2 = map(int,test['grt'][target].numpy())
    img_rgb = cv.imread(target_fnm)
    img_rgb = cv.rectangle(img_rgb,(x1,y1),(x2,y2),color=[0,0,255],thickness=2)

    if trans:
        rects_ = test_g_rect.clone()[target]
    else:
        rects_ = test['p_rect'][target]
    rects,scores = [],[]
    for (i,j) in zip(rects_, output_raw[target]):
        if torch.argmax(j) == 1:
            #px1, py1, px2, py2 = i.numpy()
            probs = torch.softmax(j,dim=0).numpy()
            if probs[1]>=0.6:
                rects.append(i.numpy())
                scores.append(probs[1])
    if len(rects)>0:
        best_rect, best_score = nms(rects,scores)
        img_rgb = cv.rectangle(img_rgb, (best_rect[0], best_rect[1]), (best_rect[2], best_rect[3]), color=[0, 255, 0], thickness=1)
        cv.putText(img_rgb, '"%d" : %.4f'%(int(os.path.split(target_fnm)[-1][:-4]), best_score), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.imshow(str(target), img_rgb)

##
