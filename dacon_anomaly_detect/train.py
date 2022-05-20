import argparse
from torchvision import models
import warnings
warnings.filterwarnings('ignore')
from dacon_anomaly_detect.utils import *
import gc
from tqdm import tqdm as tqdm
from sklearn.metrics import classification_report
##
parser = argparse.ArgumentParser(description='parameters')

# train params
parser.add_argument('--resize',         default=768,     type=int,        help = 'resize') #@@@@@@@@@@@@@@@@@@@@@@@@@ 768??
parser.add_argument('--lr',             default=1e-4,    type=float,    help = '')  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 1e-4로 조정 예정
parser.add_argument('--wd',             default=5e-2,   type=float,    help = '')

parser.add_argument('--batch_size',     default=16,     type=int,      help = '')
parser.add_argument('--max_epochs',     default=30,     type=int,      help = 'number of epochs')
parser.add_argument('--n_gpu',          default=2,      type=int, help = 'save_dir')

# preprocess params
parser.add_argument('--min_num',        default=100,     type=int,        help = 'oversample limit')

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# pl result directory
parser.add_argument('--sdir',           default=r'C:\Users\82109\PycharmProjects\object-detection\dacon_anomaly_detect', type=str, help = 'save_dir')
parser.add_argument('--name',           default='log',  type=str, help = 'log')
# pl params
parser.add_argument('--lr_patience',    default=5,       type=int,    help = '')
parser.add_argument('--es_patience',    default=20,      type=int,    help = '')

parser.add_argument('--pretrain',       default=False,   type=int,      help = 'use pretrained weight')
parser.add_argument('--labeling',       default='flat',  type=str,      help = '{flat, sep} whether seperate class and state')
parser.add_argument('--test_show',      default=False,    type=int,      help = 'show intermediate process by figure')
parser.add_argument('--random_seed',    default=42,      type=int,      help = '')
hparams = parser.parse_args(args=[]) # 테스트용

## 1. get raw data
rdir = r'D:\dacon\open'
trnx_, valx, trny_, valy, tools, tstx = split(rdir, 0.25)

trnx,trny = oversample(trnx_,trny_,min_num=hparams.min_num)

# alb = A.Compose([A.Resize(hparams.resize,hparams.resize),
#                  A.Affine(),
#                  A.Normalize(),
#                  ToTensorV2()])

# version 5
alb = A.Compose([A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.Transpose(p=0.5),
                 A.Rotate(15, border_mode=cv.BORDER_CONSTANT, mask_value=.0, p=0.5),
                 A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                 A.Resize(hparams.resize,hparams.resize),
                 A.Normalize(),
                 ToTensorV2()])

trnset = custom_dset(trnx,trny,trans=alb,train=True,resize=hparams.resize)
valset = custom_dset(valx,valy,trans=alb,train=True,resize=hparams.resize)

trn_loader = DataLoader(trnset, batch_size=hparams.batch_size ,shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(valset, batch_size=hparams.batch_size ,shuffle=False, drop_last=True, num_workers=0, pin_memory=True)

## 3. configure data
# if hparams.test_show:
#     confine(classes, states, numbers, numbers_aug, i2c, labeling= hparams.labeling)

# class rgb(Dataset):
#     def __init__(self,trnx):
#         super().__init__()
#         self.trnx = trnx
#     def __len__(self):
#         return len(trnx)
#     def __getitem__(self, item):
#         img = cv.imread(self.trnx[item]) / 255
#         img = torch.FloatTensor(img)
#         rgb_m = img.mean(dim=[0,1])
#         rgb_s = img.std(dim=[0,1])
#         return {'mean':rgb_m,
#                 'std' :rgb_s}
#
# rgb_loader = DataLoader(rgb(trnx), batch_size=256, drop_last=False, shuffle=False)
# rgb_m = torch.Tensor([])
# rgb_s = torch.Tensor([])
# for i in tqdm(rgb_loader):
#     rgb_m = torch.cat([rgb_m,i['mean']])
#     rgb_s = torch.cat([rgb_s,i['std']])
# rgb_m.mean(dim=0)
# rgb_s.mean(dim=0)

## test aug

# mean, std는 사전에 확인
# alb = A.Compose([A.Resize(hparams.resize,hparams.resize, interpolation=cv.INTER_AREA),
#                  #A.ColorJitter(brightness=.0, contrast=.2, saturation=.2, hue = .2, p=.5),
#                  #A.Rotate(limit=10, p = 0.5, border_mode=cv.BORDER_CONSTANT, value=0),
#                  # A.OneOf([A.HorizontalFlip(p=.5),
#                  #          A.VerticalFlip(p=.5)]),
#                  A.Normalize(),
#                  ToTensorV2(),
#                  #A.Normalize(mean=[0.4010, 0.4084, 0.4344],std=[0.1757, 0.1850, 0.1871]),
#                  ],)
#
# if hparams.test_show:
#     #
#     n=40
#     img_raw = cv.imread(f'{trnx[n]}')
#     img = alb(image=img_raw)
#     cv.imshow('before',img_raw)
#     cv.imshow('after',img['image'].permute(1,2,0).numpy()*.18+0.45)
##
# base = models.resnet18(pretrained=True)

##
callbacks = get_callbacks(hparams)
print('Call trainer...')
trainer = Trainer(callbacks=callbacks['callbacks'],
                  max_epochs=hparams.max_epochs,
                  gpus=hparams.n_gpu,
                  precision=16,
                  logger=callbacks['tube'],
                  deterministic=True, accelerator='dp', accumulate_grad_batches=2)
print('Train model...')
model = net(hparams)

trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

checkpoint_callback = callbacks['callbacks'][0]
best_pth = checkpoint_callback.kth_best_model_path

##
fig = plt.figure(figsize=[18,6])
n = 2
ax1_leg,ax2_leg = [],[]
for i in range(n,0,-1):
    best_pth = glob(r'C:\Users\82109\PycharmProjects\object-detection\dacon_anomaly_detect\log\*/*/*.ckpt')[-i]
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

# model = classifier(hparams)
model = net(hparams)
model = model.load_from_checkpoint(best_pth)
model.eval()
model.cuda()
print(best_pth)
#
tstset = custom_dset(tstx, [0]*len(tstx), trans=alb,train=False,resize=hparams.resize)
tst_loader = DataLoader(tstset, batch_size=16, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
pred_c,pred_s,target_c,target_s = [],[],[],[]
for v in tqdm(tst_loader):
    with torch.no_grad():
        output = model({'img':v['img'].cuda()})
        out_c = output.argmax(dim=-1).tolist()
        pred_c.extend(out_c)

frame = pd.DataFrame([tools['i2cs'][i] for i in pred_c])

ans = pd.read_csv(r'D:\dacon\open/sample_submission.csv')
ans['label'] = frame
ans.to_csv(r'%s/../../result.csv'%best_pth,index=False)

#################################################################################################################################################### 검증

val_loader_ndrop = DataLoader(valset, batch_size=16 ,shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

pred_c = []
for v in tqdm(val_loader_ndrop):
    with torch.no_grad():
        output = model({'img':v['img'].cuda()})
        out_c = output.argmax(dim=-1).tolist()
        pred_c.extend(out_c)

frame = pd.DataFrame([tools['i2cs'][i] for i in pred_c])

target_c = []
for i in tqdm(range(valset.__len__())):
    target_c.append(valset[i]['label_c'].item())



# Acc : (np.array(pred_c)==np.array(target_c)).sum()/len(pred_c)


print(classification_report(target_c,pred_c))
score_function(target_c,pred_c) # F1 score
tools['i2cs'][6]
tools['i2cs'][27]
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 6, 27 upsamp??

pred = pd.DataFrame([tools['i2cs'][i] for i in pred_c])
target = pd.DataFrame([tools['i2cs'][i] for i in target_c])

target = np.array(target_c)
pred = np.array(pred_c)
crosstab = pd.crosstab(target,pred,rownames=['target'],colnames=['pred'])

for i in np.arange(88):
    if i not in crosstab.index:
        crosstab.loc[i] = 0
    if i not in crosstab.columns:
        crosstab.loc[:,i] = 0

crosstab.sort_index(inplace=True)
crosstab = crosstab[np.arange(88)]
plt.figure();plt.imshow(crosstab)

good_ind = np.arange(88)[np.array([i.split('-')[-1] for i in tools['cs2i']])=='good']
#plt.imshow(crosstab.loc[good_ind,good_ind])




##