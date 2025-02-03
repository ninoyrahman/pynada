import sys;sys.path.append('/u/nrahman/workspace/nada/branches/pynada/')
import matplotlib as mpl;import matplotlib.pyplot as plt;import numpy as np;import h5py
#mpl.rcParams['text.usetex']='True'
mpl.rcParams['axes.grid'] = True;mpl.rcParams['axes.labelsize']=20;mpl.rcParams['lines.linewidth']=3;mpl.rcParams['figure.figsize'] = 18,10

from plotter import plotter;tp=plotter();hp=plotter()
from analysis import analysis;ta=analysis();ha=analysis()
model='c115e';dir='/ptmp/nrahman/collapse/'+model+'/data3D/'
hp.read(dir);ha.read(dir);hp.touch();ha.touch()
tp.read(dir+'neutrinos');ta.read(dir+'neutrinos');tp.touch();ta.touch()

ts=394+ha.timesteps
te=394+ha.timesteps
#t_b = ha.gettime(26,26)
t_bh=ha.gettime(396,396)
time=ha.gettime(te,te)
rmin=0
rmax=1000

rho=ha.readvar('rho',ts,te).mean(axis=(1,2)).T
temp=ha.readvar('temp',ts,te).mean(axis=(1,2)).T
vr=ha.readvar('velocity/velx',ts,te)
vra=vr.mean(axis=(1,2)).T
alp=ha.readvar('alp',ts,te).mean(axis=(1,2)).T
ve=np.sqrt(1.0-alp*alp)
Ltot=ta.calculate_total_luminosity(ts-1,te-1)/1e51
r=ha.calculate_radial_distance(ts,te).mean(axis=(1,2)).T/1e5
rah=ha.readvar('AH/rad1d',ts,te).mean(axis=(1,2)).T/1e5
x1,y1=[rah[0],rah[0]],[-0.5,0.5]

fig=plt.figure(1)
fig.add_subplot(2,2,1)
plt.plot(r,rho)
plt.xlim([rmin,rmax]);plt.yscale('log');plt.xlabel('r [km]');plt.ylabel('rho [gm/cm3]')
fig.add_subplot(2,2,2)
plt.plot(x1,y1,color='g');plt.plot(r,vra,color='k');plt.plot(r,vr.T[:,:,0,-1],linewidth=0.1,color='b');plt.plot(r,ve,color='r')
plt.xlim([rmin,rmax]);plt.ylim([-0.5,0.5]);plt.xlabel('r [km]');plt.ylabel('vr [c]')
fig.add_subplot(2,2,3)
plt.plot(r,temp)
plt.xlim([rmin,rmax]);plt.xlabel('r [km]');plt.ylabel('T')
fig.add_subplot(2,2,4)
plt.plot(r,Ltot[0,:,:])
plt.ylim([0,300]);plt.xscale('log');plt.xlabel('r [km]');plt.ylabel('L [erg/s]')
plt.suptitle(model+' tpbh='+str(time-t_bh).replace('[','').replace(']','')+'s',fontsize=22)
plt.subplots_adjust(hspace=0.4)

plt.show()
