import matplotlib.pyplot as plt
from face_landmark.utils import *
from albumentations.pytorch.transforms import ToTensorV2
from facenet_pytorch import MTCNN
##
seed_everything(seed=42)

## dset
fdir = r'D:\cv\Dataset\300w\300W\01_Indoor'
fnms = glob('%s/*png'%fdir)[:-1]
fnms_ano = [i.replace('.png','.pts',1) for i in fnms]


trnx,tstx,trny,tsty = train_test_split(fnms,fnms_ano,test_size=0.1968)
## Compare best known models
img = custom(trnx,trny,size=512,pretrained=None,key=None,trans=None,train=False)[30]['img']

# dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\82109\PycharmProjects\object-detection\todo\todo_face_landmark\shape_predictor_68_face_landmarks.dat")
rects = detector(img,1)
if len(rects)>0:
    shape = predictor(img,rects[0])
    for i in range(68):
        x,y = shape.part(i).x,shape.part(i).y
        img = cv.circle(img,(x,y),1,[0,255,0],-1)
    img = cv.rectangle(img,(rects[0].left(),rects[0].top()),(rects[0].right(),rects[0].bottom()),[0,255,0],1)
    img = cv.putText(img, 'dlib', (rects[0].left(),rects[0].top()),0, 0.5, [0, 255, 0], thickness=2)

# mtcnn
mtcnn = MTCNN(image_size=512,keep_all=True)
bbox,conf,landmark = mtcnn.detect(img,landmarks=True)
for e,i in enumerate(landmark[0]):
    img = cv.circle(img,(int(i[0]),int(i[1])),2,[255,0,0],thickness=-1)
    img = cv.putText(img, str(e), (int(i[0]), int(i[1])),0,0.3, [255, 0, 0])

x1,y1,x2,y2=bbox[0].astype(int)
img = cv.rectangle(img,(x1,y1),(x2,y2),[255,0,0],1)
img = cv.putText(img, 'mtcnn', (x1,y1),0,0.5, [255, 0, 0], thickness=2)

cv.imshow('1',img[...,::-1])
## fine tuning (using FPN backbone) : based on PWC models

trans = A.Compose([A.Resize(224,224),
                   A.Normalize(),
                   ToTensorV2()])

pretrained = models.resnet18(pretrained=True)
pretrained.eval()
pretrained.requires_grad_(False)

key = 'layer2,layer3,layer4'.split(',')
trn_set = custom(trnx,trny,size=224,pretrained=pretrained,key=key,trans=trans,train=True)
tst_set = custom(tstx,tsty,size=224,pretrained=pretrained,key=key,trans=trans,train=True)

trn_loader = DataLoader(trn_set,batch_size=37,drop_last=True,shuffle=True)
tst_loader = DataLoader(tst_set,batch_size=37,drop_last=True,shuffle=False)

##
sdir = r'C:\Users\82109\PycharmProjects\object-detection\todo\todo_face_landmark'

callbacks = get_callback(sdir)

print('Call trainer...')
trainer = Trainer(callbacks=callbacks['callbacks'],
                  max_epochs=50,
                  gpus=2,
                  logger=callbacks['tube'],
                  deterministic=True, accelerator='dp', accumulate_grad_batches=2)
print('Train model...')

model = pwc()
trainer.fit(model, trn_loader, tst_loader)
print('\t\tFitting is end...')

checkpoint_callback = callbacks['callbacks'][0]
best_pth = checkpoint_callback.kth_best_model_path
##

sdir = r'C:\Users\82109\PycharmProjects\object-detection\todo\todo_face_landmark'
best_pth = glob(r'%s\log\*/*/*.ckpt'%sdir)[-1]
print(best_pth)

fig = plt.figure(figsize=[18,6])
result = pd.read_csv(r'%s/../../metrics.csv'%best_pth)
ax1 = fig.add_subplot(111)
result.groupby('epoch').mean().plot(y=['trn_loss','val_loss'],ax=ax1,marker='*')

model = pwc()
model = model.load_from_checkpoint(best_pth)
model.eval()

##
n = 3
score_based_ths = 0.9
best_based_ths = 0.00
phase='valid'

total_pred,total_dlib,total_mtcnn = 0,0,0
singles = [1,2,3,4,6,7,8,11,12,15,18,20,23,24,26,28,29,30,31,36]

#for n in range(len(tst_set)):
for n in singles:
    ##
    n=28
    if phase=='train':
        mat = trn_set[n]
        img = cv.imread(trnx[n])
    else:
        mat = tst_set[n]
        img = cv.imread(tstx[n])
    for k in mat.keys():
        mat[k] = mat[k].unsqueeze(0)

    xx,yy = np.meshgrid(np.arange(224),np.arange(224))
    img0 = cv.resize(img,(224,224))
    img1 = img0.copy()
    img2 = img1.copy()
    img3 = img2.copy()

    with torch.no_grad():
        out = model(mat)
    out_c,out_xy = out[...,:69],out[...,69:]
    score,cls = torch.max(nn.Softmax(dim=-1)(out_c.squeeze()),dim=-1)

    # By Score : TODO
    # ind = (score>score_based_ths) & (cls!=0)
    # print('Score Based Detected:',ind.sum())
    # for x,y in zip(xx[ind],yy[ind]):
    #     img = cv.circle(img,(x,y),1,[0,0,255],-1)

    # By best points
    pred_cxy = {i:[] for i in range(1,69)}
    for i in range(1,69):
        best_score = score[cls==i]
        if len(best_score)>0:
            best_p = best_score.argmax()
            if best_score[best_p]>best_based_ths:
                cxy=[xx[cls == i][best_p], yy[cls == i][best_p]]
                pred_cxy[i]=np.array(cxy)
                img1 = cv.circle(img1,cxy,1,[0,255,255],-1)

    # mtcnn
    mtcnn = MTCNN(image_size=224,keep_all=True)
    bbox,conf,landmark = mtcnn.detect(img2,landmarks=True)

    mtcnn_cxy = {i:[] for i in range(1,69)}
    mtcnn_key = [38, 45, 34, 49, 55]
    try:
        for lm in landmark:
            for e,i in enumerate(lm):
                img2 = cv.circle(img2,(int(i[0]),int(i[1])),2,[0,0,255],thickness=-1)
                mtcnn_cxy[mtcnn_key[e]] = np.array([i[0],i[1]])
    except:
        pass
    print('dlib...')
    dlib_cxy = {i:[] for i in range(1,69)}
    rects = detector(img3,1)
    if len(rects)>0:
        shape = predictor(img3,rects[0])
        for i in range(68):
            x,y = shape.part(i).x,shape.part(i).y
            img3 = cv.circle(img3,(x,y),1,[255,255,0],-1)
            dlib_cxy[i+1] = np.array([x,y])

    # grt
    grt = {i:[] for i in range(1,69)}
    for x,y in zip(xx[mat['ano'][0,0]!=0],yy[mat['ano'][0,0]!=0]):
        grt_c = mat['ano'][0, 0, y, x].item()
        grt[grt_c] = np.array([x,y])
        img0 = cv.circle(img0, (x, y), 1, [0, 255, 0], -1)
        # img1 = cv.circle(img1, (x, y), 1, [0, 255, 0], -1)
        # img2 = cv.circle(img2, (x, y), 1, [0, 255, 0], -1)
        # img3 = cv.circle(img3, (x, y), 1, [0, 255, 0], -1)
        #img = cv.putText(img, str(int(grt_c)), (x,y), 0, 0.2, [0, 255, 0])

    cv.imshow('0', cv.resize(img0, (600, 600)))
    cv.imshow('1',cv.resize(img1,(600,600)))
    cv.imshow('2',cv.resize(img2,(600,600)))
    cv.imshow('3',cv.resize(img3,(600,600)))
    cv.imwrite(r'C:\Users\82109\PycharmProjects\object-detection\todo\todo_face_landmark\log\version_0\fig/grt.jpg',cv.resize(img0, (600, 600)))
    cv.imwrite(r'C:\Users\82109\PycharmProjects\object-detection\todo\todo_face_landmark\log\version_0\fig/pwc.jpg',cv.resize(img1, (600, 600)))
    cv.imwrite(r'C:\Users\82109\PycharmProjects\object-detection\todo\todo_face_landmark\log\version_0\fig/mtcnn.jpg',cv.resize(img2, (600, 600)))
    cv.imwrite(r'C:\Users\82109\PycharmProjects\object-detection\todo\todo_face_landmark\log\version_0\fig/dlib.jpg',cv.resize(img3, (600, 600)))




    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ERROR
    d = 224
    err_pred,err_dlib = [],[]
    err_mtcnn,err_pred_m,err_dlib_m = [],[],[]
    for k in grt.keys():
        if len(grt[k])==0:
            continue
        if len(pred_cxy[k])==0:
            err_pred_ = 1
        else:
            err_pred_ = abs(grt[k]-pred_cxy[k]).sum()/d
        if len(dlib_cxy[k])==0:
            err_dlib_ = 1
        else:
            err_dlib_ = abs(grt[k]-dlib_cxy[k]).sum()/d

        err_pred.append(err_pred_*100)
        err_dlib.append(err_dlib_*100)
        if len(mtcnn_cxy[k])!=0:
            err_mtcnn_ = abs(grt[k] - mtcnn_cxy[k]).sum() / d
        else:
            err_mtcnn_ = 1
        err_mtcnn.append(err_mtcnn_*100)
        err_pred_m.append(err_pred_*100)
        err_dlib_m.append(err_dlib_*100)



    print(f'NME               : pred {np.mean(err_pred).round(2)}, dlib {np.mean(err_dlib).round(2)}')
    if len(err_pred_m)>0:
        print(f'NME(MTCNN points) : pred {np.mean(err_pred_m).round(2)}, dlib {np.mean(err_dlib_m).round(2)}, mtcnn {np.mean(err_mtcnn).round(2)}')
    else:
        print('MTCNN ERROR')

    total_pred+=np.mean(err_pred_m).round(2)
    total_dlib+=np.mean(err_dlib_m).round(2)
    total_mtcnn+=np.mean(err_mtcnn).round(2)

total_pred/=len(tst_set)
total_dlib/=len(tst_set)
total_mtcnn/=len(tst_set)
print(f'PRED {total_pred}, DLIB {total_dlib}, MTCNN {total_mtcnn}')


