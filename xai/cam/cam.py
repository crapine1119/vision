import torch
from torch import nn
from torch.optim import optimizer as opt
from torchvision import transforms
from torchvision import models
from torch import nn
import cv2 as cv
from PIL import Image as pil
import matplotlib.pyplot as plt
## Image load
img_raw = cv.imread(r'D:\cv\Dataset\coco_2017\val2017/000000000285.jpg',cv.IMREAD_COLOR)
img_raw = cv.cvtColor(img_raw,cv.COLOR_BGR2RGB)
#
trans = transforms.Compose([transforms.Resize([224,224]),
                            transforms.ToTensor(),
                            transforms.Normalize(0.5,0.5)])

img = trans(pil.fromarray(img_raw)).unsqueeze(0) # {1,3,224,224}, normalized
## networks
class squeeze(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.squeeze()
#
class vgg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg11(pretrained=True)
        self.net = self.model.features[:20]
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                squeeze(),
                                nn.Linear(512,num_classes,bias=True),
                                nn.Softmax())
    def get_conv_result(self, net, x):
        return torch.squeeze(net(x))
    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x.squeeze()
#
##
num_classes = 10
model = vgg(num_classes = num_classes)
model.eval()
output = model(img)

# output of last conv (features)
features = model.get_conv_result(model.net,img).detach() # 512,14,14

# weight of fc (class, last channel)
class_weight = model.state_dict()['fc.2.weight'] # 10,512

# multiply feature map & weight of each class
feature_mul = (features.unsqueeze(0)*
               class_weight.view(10,512,1,1)).sum(1)

# upsample feature_mul to train image
upsample = nn.Upsample(scale_factor=img.shape[-1]/features.size()[-1], mode='bilinear')

# cam result of all classes
cam_result = upsample(feature_mul.unsqueeze(0))[0]
##
for class_no in range(num_classes):
    plt.figure(figsize=[9,9])
    unnorm = (img[0].permute(1, 2, 0)*.5 +.5).numpy()
    plt.imshow(unnorm, alpha=.5)
    #
    cam_min,cam_max = cam_result[class_no].min(),cam_result[class_no].max()
    final = (cam_result[class_no]-cam_min)/(cam_max-cam_min)
    c = plt.imshow(final.numpy(), cmap='jet',alpha=.3)
    plt.colorbar(c)