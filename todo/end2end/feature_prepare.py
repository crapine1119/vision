import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from todo.end2end.utils import *
## Imagenet Image load
seed_everything()

#fdir = r'D:\cv\Dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
fdir = 'D:\cv\Dataset\Imagenet'

alb = A.Compose([A.Resize(224,224),
                 A.Normalize(),
                 ToTensorV2()],
                bbox_params=A.BboxParams(format='pascal_voc',
                                         label_fields=['category_ids']),)

# trnset = voc_dataset(fdir, phase='train', several=True, trans=alb, imshow=True)
trnset = imagenet(fdir, stride=4, trans=alb, several=False)
##
sdir = r'D:\cv\Dataset\Imagenet'
for i in tqdm(range(len(trnset))):
    try:
        mat = trnset[i]
        name = trnset.trn_files[i][-13:-5]
        torch.save(mat['imgs'],'%s/img_tensor/feat_%s.pt'%(sdir,name))
        torch.save(mat['labels_mat'], '%s/label_tensor/targ_%s.pt' % (sdir, name))
        del mat
        gc.collect()
    except:
        pass
