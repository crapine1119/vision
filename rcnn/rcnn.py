

from object_detection.rc import *
##
seed_everything(seed=42)

root_dir = r'D:\cv\Dataset/coco_2017/val2017/'
ano_fnm = r'D:\cv\Dataset/coco_2017/annotations/instances_val2017.json'

image_list, id2ctg, ctg2name = get_items(ano_fnm)

ctg2id = {list(id2ctg.values())[i]:list(id2ctg.keys())[i] for i in range(len(id2ctg))}

trn_x, tst_x =  train_test_split(image_list,test_size=0.4)
vld_x = tst_x[:len(tst_x)//2]
## extract car(3) dataset
min_num = 5

def do_ss(min_num,dict):
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
    trn_imgs, trn_labels = [], []
    for e,box in enumerate(rects):
        if flag==0: # 2000개 까지만으로 제한
            roi = list(box)
            roi[2] += roi[0]
            roi[3] += roi[1]
            iou = get_iou(roi,grt)

            if count_t < min_num:
                if iou>=0.1:
                    #cut = img_rgb[roi[1]:roi[3],roi[0]:roi[2]]
                    #resized = cv.resize(cut, (224,224), interpolation= cv.INTER_AREA)
                    #trn_imgs.append(resized)
                    trn_imgs.append(roi)
                    trn_labels.append(dict['labels'].item())
                    count_t +=1
            else:
                fflag = 1
            #
            if count_f < min_num*2:
                if iou<0.1:
                    # cut = img_rgb[roi[1]:roi[3], roi[0]:roi[2]]
                    # resized = cv.resize(cut, (224, 224), interpolation=cv.INTER_AREA)
                    # trn_imgs.append(resized)
                    trn_imgs.append(roi)
                    trn_labels.append(0)
                    count_f+=1
            else:
                bflag = 1
        if fflag==1 and bflag==1:
            break
    return np.array(trn_imgs, dtype=np.int_), np.array(grt), np.array(trn_labels, dtype=np.int_)

def sep_pn(min_num=8,when=50,flst=trn_x,sdir=r'D:\cv\01rcnn\sample',make_sample=False):
    if os.path.isdir(sdir)==False:
        os.makedirs(sdir)
    count=0
    for dict in flst:
        if count==when:
            break
        if dict['labels']==3:
            print('Do selective search')
            rects,grt,labels = do_ss(min_num,dict)
            if (labels!=0).sum()>=min_num:
                count += 1
                print(dict['labels'].item(),count)
                if make_sample:
                    fnm = '%s/%012d.jpg' % (root_dir, dict['image_id'])
                    img_rgb = cv.imread(fnm)
                    # img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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

#sep_pn(8,50,trn_x,sdir,make_sample=True)
#sep_pn(8,25,vld_x,r'D:\cv\01rcnn\sample_valid',make_sample=True)
#sep_pn(8,25,tst_x,r'D:\cv\01rcnn\sample_test',make_sample=True)


## customizing
import os

ex_fnm = glob.glob(r'%s/*p.csv'%(sdir))[0]
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
a = customdataset(sdir,ctg2id=ctg2id)
b = customdataset(sdir+'_valid',ctg2id=ctg2id)
c = customdataset(sdir+'_test',ctg2id=ctg2id)

trn_loader = DataLoader(a, batch_size=1, shuffle=True, drop_last= True)
vld_loader = DataLoader(b, batch_size=1, shuffle=True, drop_last= True)
tst_loader = DataLoader(c, batch_size=c.__len__(), shuffle=False, drop_last= False)
##
pretrained = nn.Sequential(*[*models.resnet50(pretrained=True).children()][:-1])
pretrained.cuda().eval()
for i in pretrained.parameters():
    i.requires_grad_(False)


model = nn.Linear(2048,2) # 15(N) * 2(binary)
model.cuda() # 반드시 가장 나중에

real_batch = a.positive_sizes[0]+a.negative_sizes[0]
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)

max_epoch = 30
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
                feat = pretrained(input)
                output = model(feat.squeeze())
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
                'Mean Acc.': '{:04f}'.format(acc['train'] / ((idx + 1) * real_batch)),
                'Mean Val Loss': '{:06f}'.format(loss['valid'] / (idx_v + 1)),
                'Mean Val Acc.': '{:04f}'.format(acc['valid'] / ((idx_v + 1) * real_batch))})
    return  loss['train']/batch_iter, acc['train']/(batch_iter*real_batch), loss['valid']/batch_iter_v, acc['valid']/(batch_iter_v*real_batch)

total_loss, total_loss_v = [], []
total_acc, total_acc_v = [], []
for epoch in range(max_epoch):
    loss, acc, loss_v, acc_v = cl(epoch,trn_loader,vld_loader)
    #
    total_loss.append(loss.detach().item())
    total_loss_v.append(loss_v.item())
    total_acc.append(acc.item())
    total_acc_v.append(acc_v.item())
plt.figure()
plt.plot(total_loss);plt.plot(total_loss_v);plt.title('Loss')
plt.figure()
plt.plot(total_acc);plt.plot(total_acc_v);plt.title('Acc')

##
test = next(iter(tst_loader))

from sklearn.metrics import f1_score
def acc(pred,real):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

feat = pretrained(test['img'].view(-1,3,224,224).cuda())
model.eval()
pred = model(feat.squeeze())

test['grt'].size()

print('ACC:',acc(pred,test['label'].view(-1)))

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

in_features = 2048
out_features = 4
bbox_regressor = nn.Linear(in_features, out_features)
bbox_regressor.cuda()
def rg(max_epoch,pretrained,bbox_regressor,trn_loader):
    batch_iter = trn_loader.__len__()
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
            p_rect = dict['p_rect'][0].cuda() # lt,rb
            grt = dict['grt'][0].cuda()       # lt,rb
            target = calcul(p_rect,grt)
            #
            features = pretrained(input[label!=0]) # positive feature만
            # zero the parameter gradients
            optimizer.zero_grad()
            output = bbox_regressor(features.squeeze())
            batch_loss = criterion(output,target)
            batch_loss.backward()
            optimizer.step()
            loss+=batch_loss*target.size(0)
        total_loss.append(loss/(batch_iter))
        print('\t Epoch : %02d, Loss : %.4f'%(epoch,(loss/(batch_iter)).item()))
    return total_loss
    # regressor

box_loss = rg(30,pretrained,bbox_regressor,trn_loader)

##
test['img'].size()

model.eval()
bbox_regressor.eval()
with torch.no_grad():
    features = pretrained(test['img'].view(-1, 3, 224, 224).cuda())
    output = model(features.squeeze())
    tset = bbox_regressor(features.squeeze())

output = output.view(20,16,2)[:,:8] # label prob
tset = tset.view(20,16,4)[:,:8]     # 변환하기 위한 가중치 t

target_label = test['label'][:,:8] # 20,16(~8:True)
test['grt'].size()

_,ind = torch.max(output,dim=-1)

x_min = test['p_rect'][:,:,0]
y_min = test['p_rect'][:,:,1]
x_max = test['p_rect'][:,:,2]
y_max = test['p_rect'][:,:,3]

pw = x_max - x_min
ph = y_max - y_min
px = x_min + pw/2
py = y_min + ph/2

tx = tset[:,:,0].detach().cpu()
ty = tset[:,:,1].detach().cpu()
tw = tset[:,:,2].detach().cpu()
th = tset[:,:,3].detach().cpu()

gx = (pw*tx + px).unsqueeze(2) # 중앙x
gy = (ph*ty + py).unsqueeze(2) # 중앙y
gw = (pw*torch.exp(tw)).unsqueeze(2) # 전체 w
gh = (ph*torch.exp(th)).unsqueeze(2) # 전체 h

test_g_rect = torch.cat([gx-gw/2,
                         gy-gh/2,
                         gx+gw/2,
                         gy+gh/2],dim=2)

##
for i in range(10):
    target_fnm = glob.glob(r'D:\cv\01rcnn\sample_test/*.jpg')[i]
    x1,y1,x2,y2 = map(int,test['grt'][i].numpy())
    img = cv.imread(target_fnm)
    img = cv.rectangle(img,(x1,y1),(x2,y2),color=[0,0,255],thickness=2)


    for bb,bb_r,l in zip(test['p_rect'][i],test_g_rect[i],ind[i]):
        if l==1: # 모델이 물체라고 예측한 박스들만
            x1,y2,x2,y2 = bb.numpy()
            img_b = cv.rectangle(img, (x1, y1), (x2, y2), color=[0, 255, 0], thickness=1)
            rx1,ry1,rx2,ry2 = bb_r.int().numpy()
            img_a = cv.rectangle(img, (rx1, ry1), (rx2, ry2), color=[0, 255, 0], thickness=1)

    cv.imshow('%s_before'%i,img_b)
    cv.imshow('%s_after'%i, img_a)