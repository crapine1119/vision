import pandas as pd
import numpy as np
from glob import glob as glob
import os
from PIL import Image as pil
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import open3d as o3d
##
fdir = r'D:\cv\Dataset\soslab_samples'

cam_dir = glob('%s/*cam*'%fdir)


json_dir = glob('%s/*json/*'%fdir)
lidar_dir = glob('%s/*lidar/lidar/*'%fdir)
lidar_label_dir = glob('%s/*lidar/lidar_label/*'%fdir)
##
cam1 = cam_dir[0]
calb,cam,cam_label = os.listdir(cam1)


imgs = glob(f'{cam1}/{cam}/*')
##
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
camera = Camera(fig)

for img_path in imgs[:30]:
    label_path = glob(f'{cam1}/{cam_label}/*{os.path.basename(img_path)[:-4]}*')[0]

    image = cv.imread(img_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # pil.fromarray(image).show()

    label = pd.read_csv(label_path,header=None,delim_whitespace=True)
    label.columns = 'id,type,cx,cy,w,h,difficulty'.split(',')
    #

    ax.imshow(image)

    rxs = label['cx']-label['w']/2
    rys = label['cy']-label['h']/2
    rws = label['w'].copy()
    rhs = label['h'].copy()

    for i in range(len(label)):
        ax.add_patch(
            ptc.Rectangle(
                (rxs[i],rys[i]),rws[i],rhs[i],fill=False,ec='g'))
    camera.snap()
ani = camera.animate(interval=50,blit=True)
##
lidar_fnm = lidar_dir[0]
with open(lidar_fnm,'rb') as file:
    p_lidar = np.fromfile(file,dtype=np.float32)
    #p_lidar = file.read()

points = p_lidar.reshape((-1,4))[:,:3]

# 그냥 그리는 법
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(points[:,0],points[:,1],points[:,2],'.',ms=0.5)

# pcd로 그리는 법
o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
o3d.visualization.draw_geometries([o3d_pcd])

label_lidar = pd.read_csv(lidar_label_dir[0],delim_whitespace=True,header=None)
##

