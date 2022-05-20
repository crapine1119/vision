import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as sts
from sklearn.cluster import DBSCAN as dbscan
from sklearn.cluster import KMeans as kmean
import hdbscan
import random
import os
##
def seed_everything(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print('Seed : %s'%seed)
seed_everything(0)

dset = np.random.rand(100,5)

dset = pd.DataFrame(dset)

dset.columns = ['ID','나이','연봉','연차','성과']


dset['ID'] = np.arange(100)
dset['나이'] = (dset['나이']*50).round(2)
dset['연봉'] = dset['나이']*dset['연차']*200
dset['연차'] = np.random.rand(100,1)*4+5
scaler = sts()
dset_sts = scaler.fit_transform(dset[['연봉','연차']].values)
dset_sts = pd.DataFrame(dset_sts,columns=['연봉','연차'])

## 1. k-means
"""
center based cluster
중심점의 개수를 설정해주면, 그 중심을 기반으로 거리가 최소화되는 군집을 나눔
"""
model_km = kmean(n_clusters=5, random_state=42)
model_km.fit(dset_sts)
dset_sts['kmean'] = model_km.fit_predict(dset_sts)
print('kmean : %s'%dset_sts['kmean'].unique())

## 2. dbscan
"""
density based cluster
중심점을 정해주지 않고, 서로 밀접한 정도를 통해 군집을 나누는 방법
타원의 반경(eps)와 그 안에 있는 최소 점의 개수(min sample)을 파라미터로 가진다.
사람이 설정해야된다는 단점이 존재
"""
model_db = dbscan(eps=0.3, min_samples=5)
model_db.fit(dset_sts)
dset_sts['dbscan'] = model_db.fit_predict(dset_sts)
print('dbscan : %s'%dset_sts['dbscan'].unique())

## hdbscan
"""
hierarchical dbscan
mutual reachability (distance를 noisy robust하게 만드는 방법)를 구하고
이를 바탕으로 트리를 만든다(기준을 점점 낮춰가면서)
트리를 이용해서 클러스터를 구축하고, 알고리즘에 의해 안정적인 클러스터를 선택한다.

"""
model_hdb = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
model_hdb.fit(dset_sts)
dset_sts['hdbscan'] = model_db.fit_predict(dset_sts)
print('hdbscan : %s'%dset_sts['hdbscan'].unique())
plt.figure()
model_hdb.minimum_spanning_tree_.plot()
##
key = 'kmean,dbscan,hdbscan'.split(',')

fig = plt.figure(figsize=[18,7.5])
for f,k in zip(range(3),key):
    ax = fig.add_subplot(1,3,f+1)
    for i in dset_sts[k].unique():
        ind = dset_sts[k] == i
        ax.scatter(dset_sts['연차'][ind],
                    dset_sts['연봉'][ind],
                    label=i)

    ax.legend(loc=1)
    ax.set_xlabel('Career (year)')
    ax.set_ylabel('Income (year)')
    ax.grid('on')
    ax.set_title(k.upper(),fontsize=22)