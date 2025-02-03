import sys;sys.path.append('/u/nrahman/workspace/nada/branches/pynada/');import matplotlib as mpl;import matplotlib.pyplot as plt;import numpy as np;import h5py
mpl.rcParams['text.usetex']='True';mpl.rcParams['axes.labelsize']=20#;mpl.rcParams['figure.titlesize']=20
from plotter import plotter;tp=plotter();hp=plotter()
from analysis import analysis;ta=analysis();ha=analysis()
model='c115';dir='/ptmp/nrahman/collapse/'+model+'/data3D/'
hp.read(dir);ha.read(dir);hp.touch();ha.touch()
tp.read(dir+'neutrinos');ta.read(dir+'neutrinos');tp.touch();ta.touch()

t_b = ha.gettime(26,26)
time_tr=tp.timesteps
time_hy=hp.timesteps

vr=ha.readvar('velocity/velx',time_hy,time_hy)[0,0,:,:].T
vt=ha.readvar('velocity/vely',time_hy,time_hy)[0,0,:,:].T*ha.x[:,None]
temp=ha.readvar('temp',time_hy,time_hy)[0,0,:,:].T
ye=ha.readvar('ye',time_hy,time_hy)[0,0,:,:].T
ytot=ta.readvar('ytot',time_tr,time_tr)[:,0,:,:,:]
rho=ha.readvar('rho',time_hy,time_hy)[0,0,:,:].T
s=ha.readvar('s',time_hy,time_hy)[0,0,:,:].T
mc=ha.calculate_mass_coordinate(time_hy,time_hy)[0,:]/ha.Msolar
mcgrav=ha.calculate_mass_coordinate(time_hy,time_hy,grav=1)[0,:]/ha.Msolar
mcadm=ha.readvar('ADM/M',time_hy,time_hy).T.sum(axis=(1,2)).cumsum(axis=0)[:,0]/ha.rs
N2=ha.calculate_brunt_vaisala_frequency(time_hy,time_hy)
mass=ha.calculate_cell_mass(time_hy,time_hy)

alp=ha.readvar('alp',time_hy,time_hy)[0,0,:,:].T
phi=ha.readvar('phi',time_hy,time_hy)[0,0,:,:].T
trK=ha.readvar('trK',time_hy,time_hy)[0,0,:,:].T
bx=ha.readvar('betau/betaux',time_hy,time_hy)[0,0,:,:].T
by=ha.readvar('betau/betauy',time_hy,time_hy)[0,0,:,:].T
dy=ha.readvar('delta/deltay',time_hy,time_hy)[0,0,:,:].T
gphixx=ha.readvar('gphixx',time_hy,time_hy)[0,0,:,:].T
gphixy=ha.readvar('gphixy',time_hy,time_hy)[0,0,:,:].T
gphiyy=ha.readvar('gphiyy',time_hy,time_hy)[0,0,:,:].T
Aphixx=ha.readvar('Aphixx',time_hy,time_hy)[0,0,:,:].T
Aphixy=ha.readvar('Aphixy',time_hy,time_hy)[0,0,:,:].T
Aphiyy=ha.readvar('Aphiyy',time_hy,time_hy)[0,0,:,:].T

alpavg=ha.calculate_mass_average_quantity('alp',time_hy,time_hy)
phiavg=ha.calculate_mass_average_quantity('phi',time_hy,time_hy)
trKavg=ha.calculate_mass_average_quantity('trK',time_hy,time_hy)
bxavg=ha.calculate_mass_average_quantity('betau/betaux',time_hy,time_hy)
byavg=ha.calculate_mass_average_quantity('betau/betauy',time_hy,time_hy)
dyavg=ha.calculate_mass_average_quantity('delta/deltay',time_hy,time_hy)
gphixxavg=ha.calculate_mass_average_quantity('gphixx',time_hy,time_hy)
gphixyavg=ha.calculate_mass_average_quantity('gphixy',time_hy,time_hy)
gphiyyavg=ha.calculate_mass_average_quantity('gphiyy',time_hy,time_hy)
Aphixxavg=ha.calculate_mass_average_quantity('Aphixx',time_hy,time_hy)
Aphixyavg=ha.calculate_mass_average_quantity('Aphixy',time_hy,time_hy)
Aphiyyavg=ha.calculate_mass_average_quantity('Aphiyy',time_hy,time_hy)

vravg=ha.calculate_mass_average_quantity('velocity/velx',time_hy,time_hy)
vtavg=ha.calculate_mass_average_quantity('velocity/vely',time_hy,time_hy)*ha.x[:,None] 
tempavg=ha.calculate_mass_average_quantity('temp',time_hy,time_hy)
yeavg=ha.calculate_mass_average_quantity('ye',time_hy,time_hy)
rhoavg=ha.calculate_mass_average_quantity('rho',time_hy,time_hy)
savg=ha.calculate_mass_average_quantity('s',time_hy,time_hy)
N2m=N2*mass[:,:,:,1:ha.nx[0]-2]
N2avg=N2m.sum(axis=(1,2))/mass[:,:,:,1:ha.nx[0]-2].sum(axis=(1,2))
ytotm=ytot*mass
ytotavg=ytotm.sum(axis=(1,2))/mass.sum(axis=(1,2)) 

N2=N2[0,0,:,:].T
ytot=ytot[0,0,:,:].T
chi=ha.calculate_chi_parameter(time_hy,time_hy)
tadv=ha.calculate_advection_time_scale(time_hy,time_hy)

e_nu=ta.readvar('emean',time_tr,time_tr)[:,0,:,:,:,0]
e_anu=ta.readvar('emean',time_tr,time_tr)[:,0,:,:,:,1]
e_nx=ta.readvar('emean',time_tr,time_tr)[:,0,:,:,:,2]
em_nu=e_nu*mass
em_anu=e_anu*mass
em_nx=e_nx*mass
eavg_nu=em_nu.sum(axis=(1,2))/mass.sum(axis=(1,2))
eavg_anu=em_anu.sum(axis=(1,2))/mass.sum(axis=(1,2))
eavg_nx=em_nx.sum(axis=(1,2))/mass.sum(axis=(1,2))
e_nu=e_nu[0,0,:,:].T
e_anu=e_anu[0,0,:,:].T
e_nx=e_nx[0,0,:,:].T

L_nu=ta.readvar('Lmean',time_tr,time_tr)[0,0,0,:,:,0].T
L_anu=ta.readvar('Lmean',time_tr,time_tr)[0,0,0,:,:,1].T
L_nx=ta.readvar('Lmean',time_tr,time_tr)[0,0,0,:,:,2].T
Ltot_nu=ta.calculate_total_luminosity(time_tr,time_tr)[:,:,0]
Ltot_anu=ta.calculate_total_luminosity(time_tr,time_tr)[:,:,1] 
Ltot_nx=ta.calculate_total_luminosity(time_tr,time_tr)[:,:,2] 

se=ta.readvar('SE',time_tr,time_tr)/ta.readvar('rho',time_tr,time_tr)
sn=ta.readvar('SN',time_tr,time_tr)/ta.readvar('rho',time_tr,time_tr)
se=se[:,0,:,:,:].mean(axis=(1,2))
sn=sn[:,0,:,:,:].mean(axis=(1,2)) 

l=np.arange(0,101,2)
elat=ha.calculate_cell_lateral_kinetic_energy(time_hy,time_hy)
El=np.abs(ha.spectral_decomposition_2d(elat,101)[:,:,:300].sum(axis=(2)))
elat=elat.sum(axis=(1,2))
etot=ha.calculate_total_energy(time_hy,time_hy).sum(axis=(1,2)) 
theat=np.abs(etot/se)
 
fhydro=plt.figure(1)
fhydro.add_subplot(3,3,1)
plt.plot(ha.x/1e5,rho[:,:],linewidth=0.2);plt.plot(ha.x/1e5,rhoavg[0,:],linewidth=2,color='b') 
plt.xscale('log');plt.yscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\rho [gm/cm^3]$}');plt.grid()
fhydro.add_subplot(3,3,2)
plt.plot(ha.x/1e5,temp[:,:],linewidth=0.2);plt.plot(ha.x/1e5,tempavg[0,:],linewidth=2,color='g')
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$T [MeV]$}');plt.grid()
fhydro.add_subplot(3,3,3)
plt.plot(ha.x/1e5,ye[:,:],linewidth=0.2);plt.plot(ha.x/1e5,yeavg[0,:],linewidth=2,color='r')
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$y_e$}');plt.grid()
fhydro.add_subplot(3,3,4)
plt.plot(ha.x/1e5,ytot[:,:],linewidth=0.2);plt.plot(ha.x/1e5,ytotavg[0,:],linewidth=2,color='b')
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$y_{lep}$}');plt.grid()
fhydro.add_subplot(3,3,5)
plt.plot(ha.x/1e5,vr[:,:],linewidth=0.2);plt.plot(ha.x/1e5,vravg[0,:],linewidth=2,color='g')
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$v_r [c]$}');plt.grid()
fhydro.add_subplot(3,3,6)
plt.plot(ha.x/1e5,vt[:,:],linewidth=0.2);plt.plot(ha.x/1e5,vtavg[0,:],linewidth=2,color='r')
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$v_t [c]$}');plt.grid()
fhydro.add_subplot(3,3,7)
plt.plot(ha.x/1e5,s[:,:],linewidth=0.2);plt.plot(ha.x/1e5,savg[0,:],linewidth=2,color='b')
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$s [kb/bary]$}');plt.grid()
fhydro.add_subplot(3,3,8)
plt.plot(ha.x/1e5,mc,linewidth=2,color='g',label='rest');plt.plot(ha.x/1e5,mcgrav,linewidth=2,color='k',label='grav');plt.plot(ha.x/1e5,mcadm,linewidth=2,color='m',label='adm')
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$m [M_{\odot}]$}');plt.legend(loc='best');plt.grid()
fhydro.add_subplot(3,3,9)
plt.plot(ha.x/1e5,elat[0,:],linewidth=2,color='r')
plt.xlim([0,400]);plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$e^{lat}_{kin} [erg]$}');plt.grid()
plt.suptitle(model+' tpb='+str(ha.time-t_b).replace('[','').replace(']','')+'s',fontsize=22)
plt.subplots_adjust(hspace=0.4)

fneu=plt.figure(2)
fneu.add_subplot(3,3,1)
plt.plot(ta.x/1e5,e_nu[:,:],linewidth=0.2);plt.plot(ta.x/1e5,eavg_nu[0,:],linewidth=2,color='b') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\langle\epsilon_{\nu e}\rangle [MeV]$}');plt.grid()
fneu.add_subplot(3,3,2)
plt.plot(ta.x/1e5,e_anu[:,:],linewidth=0.2);plt.plot(ta.x/1e5,eavg_anu[0,:],linewidth=2,color='g') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\langle\epsilon_{\bar{\nu} e}\rangle [MeV]$}');plt.grid()
fneu.add_subplot(3,3,3)
plt.plot(ta.x/1e5,e_nx[:,:],linewidth=0.2);plt.plot(ta.x/1e5,eavg_nx[0,:],linewidth=2,color='r') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\langle\epsilon_{\nu x}\rangle [MeV]$}');plt.grid()
fneu.add_subplot(3,3,4)
plt.plot(ta.x/1e5,Ltot_nu[0,:]/1e51,linewidth=2,color='b') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$L_{\nu e} [10^{51} erg/s]$}');plt.grid()
fneu.add_subplot(3,3,5)
plt.plot(ta.x/1e5,Ltot_anu[0,:]/1e51,linewidth=2,color='g') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$L_{\bar{\nu} e} [10^{51} erg/s]$}');plt.grid()
fneu.add_subplot(3,3,6)
plt.plot(ta.x/1e5,Ltot_nx[0,:]/1e51,linewidth=2,color='r') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$L_{\nu x} [10^{51} erg/s]$}');plt.grid()
fneu.add_subplot(3,3,7)
plt.plot(ta.x/1e5,se[0,:],linewidth=2,color='b') 
plt.xlim([50,400]);plt.ylim([-1e21,1e21]);plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$S_E [erg/gm/s]$}');plt.grid()
fneu.add_subplot(3,3,8)
plt.plot(ta.x/1e5,sn[0,:],linewidth=2,color='g') 
plt.xlim([50,400]);plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$S_N [gm/gm/s]$}');plt.grid()
plt.suptitle(model+' tpb='+str(tp.time-t_b).replace('[','').replace(']','')+'s',fontsize=22)
plt.subplots_adjust(hspace=0.4)

fan=plt.figure(3)
fan.add_subplot(2,2,1)
plt.plot(ha.x[1:ha.nx[0]-2]/1e5,N2[:,:],linewidth=0.2);plt.plot(ha.x[1:ha.nx[0]-2]/1e5,N2avg[0,:],linewidth=2,color='b')
plt.xlim([0,400]);plt.yscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\omega_{BV}$}');plt.grid()
fan.add_subplot(2,2,2)
plt.plot(ha.x[1:ha.nx[0]-2]/1e5,chi[0,:],linewidth=2,color='g')
plt.xlim([50,400]);plt.ylim([0,1.5]);plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\chi$}');plt.grid()
fan.add_subplot(2,2,3)
plt.plot(ha.x/1e5,tadv[0,:]/theat[0,:],linewidth=2,color='r')
plt.xlim([0,400]);plt.yscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\tau_{adv}/\tau_{heat}$}');plt.grid()
fan.add_subplot(2,2,4)
plt.plot(l,El[:,0],linewidth=2,color='b')
plt.xscale('log');plt.yscale('log');plt.xlabel(r'\textit{$l$}');plt.ylabel(r'\textit{$E_l [erg]$}');plt.grid()
plt.suptitle(model+' tpb='+str(tp.time-t_b).replace('[','').replace(']','')+'s',fontsize=22)
plt.subplots_adjust(hspace=0.4)

fgr=plt.figure(4)
fgr.add_subplot(3,3,1)
plt.plot(ta.x/1e5,alp[:,:],linewidth=0.2);plt.plot(ta.x/1e5,alpavg[0,:],linewidth=2,color='b') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\alpha$}');plt.grid()
fgr.add_subplot(3,3,2)
plt.plot(ta.x/1e5,bx[:,:],linewidth=0.2);plt.plot(ta.x/1e5,bxavg[0,:],linewidth=2,color='g') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\beta_{r}$}');plt.grid()
fgr.add_subplot(3,3,3)
plt.plot(ta.x/1e5,by[:,:],linewidth=0.2);plt.plot(ta.x/1e5,byavg[0,:],linewidth=2,color='r') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\beta_{\theta}$}');plt.grid()
fgr.add_subplot(3,3,4)
plt.plot(ta.x/1e5,gphixx[:,:],linewidth=0.2);plt.plot(ta.x/1e5,gphixxavg[0,:],linewidth=2,color='b') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\bar{\gamma}_{rr}$}');plt.grid()
fgr.add_subplot(3,3,5)
plt.plot(ta.x/1e5,gphixy[:,:],linewidth=0.2);plt.plot(ta.x/1e5,gphixyavg[0,:],linewidth=2,color='g') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\bar{\gamma}_{r\theta}$}');plt.grid()
fgr.add_subplot(3,3,6)
plt.plot(ta.x/1e5,dy[:,:],linewidth=0.2);plt.plot(ta.x/1e5,dyavg[0,:],linewidth=2,color='r') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\Lambda^{\theta}$}');plt.grid()
fgr.add_subplot(3,3,7)
plt.plot(ta.x/1e5,Aphixx[:,:],linewidth=0.2);plt.plot(ta.x/1e5,Aphixxavg[0,:],linewidth=2,color='b') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\bar{A}_{rr}$}');plt.grid()
fgr.add_subplot(3,3,8)
plt.plot(ta.x/1e5,Aphixy[:,:],linewidth=0.2);plt.plot(ta.x/1e5,Aphixyavg[0,:],linewidth=2,color='g') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$\bar{A}_{r\theta}$}');plt.grid()
fgr.add_subplot(3,3,9)
plt.plot(ta.x/1e5,trK[:,:],linewidth=0.2);plt.plot(ta.x/1e5,trKavg[0,:],linewidth=2,color='r') 
plt.xscale('log');plt.xlabel(r'\textit{$r [km]$}');plt.ylabel(r'\textit{$K$}');plt.grid()
plt.suptitle(model+' tpb='+str(tp.time-t_b).replace('[','').replace(']','')+'s',fontsize=22)
plt.subplots_adjust(hspace=0.4)

#plt.tight_layout()

plt.show()
