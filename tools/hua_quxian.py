"""python3 tools/hua_quxian.py result/week3/zhuangtai.csv
曲线仅来自程序CSV，空值和重初始化处断线；阴影是模型内±2σ，不是精度认证。
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1] if len(sys.argv)>1 else 'result/week3/zhuangtai.csv')
a=np.genfromtxt(p,delimiter=',',names=True)
names=['cx','vx','cy','vy','z0','vz','yaw','omega','r0','r1','dz']
units=['m','m/s','m','m/s','m','m/s','rad','rad/s','m','m','m']
t=a['time']-a['time'][0]
fig,axs=plt.subplots(6,2,figsize=(15,17),sharex=True)
for i,(name,unit) in enumerate(zip(names,units)):
 ax=axs.flat[i];y=a[name].copy();sd=a[f'sd{i}'].copy();cuts=np.r_[False,np.diff(a['segment'])!=0];y[cuts]=np.nan;sd[cuts]=np.nan
 ax.plot(t,y,lw=1,label='EKF');ax.fill_between(t,y-2*sd,y+2*sd,alpha=.18,label='+/- 2 sigma')
 ax.set_ylabel(f'{name} ({unit})');ax.grid(alpha=.25)
axs.flat[11].plot(t,a['nis'],lw=.8);axs.flat[11].axhline(18.47,color='r',ls='--',label='default gate');axs.flat[11].set_ylabel('NIS (last accepted)');axs.flat[11].grid(alpha=.25)
for ax in axs[-1]:ax.set_xlabel('Elapsed capture-time estimate (s)')
axs.flat[0].legend();fig.suptitle('Week 3: all 11 estimated states | assumed timing/extrinsics unless verified',fontsize=13)
fig.tight_layout(rect=(0,0,1,.98));fig.savefig(p.parent/'zhuangtai_quxian.png',dpi=140);fig.savefig(p.parent/'zhuangtai_quxian.pdf');print(p.parent/'zhuangtai_quxian.png')
