'''
Created on Feb 8, 2017

@author: ninoy
'''
import h5py
import numpy as np


class eos:
    """
    class for equation of state 
    Attributes:
        filepath (str) : path to eos file
        nrho (int): number of data points in density
        ntemp (int): number of data points in temperature
        nye (int): number of data points in electron fraction
        rho (double): 1d array for density values [gm/cm3]
        temp (double): 1d array for temperature values [MeV]
        ye (double): 1d array for electron fraction values
        p (double) :: 3d array for pressure values [erg/cm3]
        eps (double) :: 3d array for internal energy values [erg/gm]
        cs2 (double) :: 3d array for sound's speed squared values [cm2/s2]
        s (double) :: 3d array for entropy values [kb/baryon]
        gamma (double) :: 3d array for gamma values
        mu_e (double) :: 3d array for electron chemical potential values [Mev]
        mu_n (double) :: 3d array for neutron chemical potential values [Mev]
        mu_p (double) :: 3d array for proton chemical potential values [Mev]
        mu_nu (double) :: 3d array for neutrino chemical potential values [Mev]
        xn (double) :: 3d array for neutron abundance values
        xp (double) :: 3d array for proton abundance values
        xa (double) :: 3d array for alpha abundance values
        xh (double) :: 3d array for heavy elements abundance values
        abar (double) :: 3d array for mean mass number of heavy elements values
        zbar (double) :: 3d array for mean proton number of heavy elements values
        energy_shift (double) :: energy_shift [erg/gm]
        qvol (double) :: quark volume fraction
    """

    def __init__(self):

        super().__init__()

        self.mb = 1.66e-24          		# baryon mass unit in g
        self.conv_MeVtoErg = 1.602e-6		# conversion factor from MeV to erg
        self.conv_ErgtoMeV = 624150.647996e0 	# conversion factor from erg to MeV

        self.filename = None
        self.nrho = None
        self.ntemp = None
        self.nye = None
        self.rho = None
        self.temp = None
        self.ye = None
        self.p = None
        self.eps = None
        self.h = None
        self.cs2 = None
        self.s = None
        self.gamma = None
        self.mu_e = None
        self.mu_n_ = None
        self.mu_p = None
        self.mu_nu = None
        self.abar = None
        self.zbar = None
        self.xn = None
        self.xp = None
        self.xa = None
        self.xh = None
        self.energy_shift = None
        self.qvol = None
        self.yp = None
        self.ypi = None
        self.ypi0 = None
        self.ypic = None
        self.ppi = None
        self.ppi0 = None
        self.upi = None
        self.upi0 = None
        self.spi = None
        self.mu_pi = None
        self.conv = None

    def readeostable(self,filename):
        """
        read eos
        Args:
            @filename (str): name of eos file
        """
        self.filename = filename
        self.__h5file=h5py.File(self.filename,'r')
        if '/pointsrho' in self.__h5file:
            self.nrho = self.__h5file['/pointsrho'][:]
        if '/pointstemp' in self.__h5file:
            self.ntemp = self.__h5file['/pointstemp'][:]
        if '/pointsye' in self.__h5file:
            self.nye = self.__h5file['/pointsye'][:]

        if '/logrho' in self.__h5file:
            self.rho = np.array(self.__h5file['/logrho'],dtype='float64')
            self.rho=10.0**self.rho[:]
        if '/logtemp' in self.__h5file:
            self.temp = np.array(self.__h5file['/logtemp'],dtype='float64')
            self.temp=10.0**self.temp[:]
        if '/ye' in self.__h5file:
            self.ye = np.array(self.__h5file['/ye'])

        if '/energy_shift' in self.__h5file:
            self.energy_shift = np.array(self.__h5file['/energy_shift'])
        if '/logpress' in self.__h5file:
            self.p = np.array(self.__h5file['/logpress'],dtype='float64')
            self.p=10.0**self.p[:]
        if '/logenergy' in self.__h5file:
            self.eps = np.array(self.__h5file['/logenergy'],dtype='float64')
            self.eps=10.0**self.eps[:]-self.energy_shift[:]

        if '/cs2' in self.__h5file:
            self.cs2 = np.array(self.__h5file['/cs2'],dtype='float64')
        if '/entropy' in self.__h5file:
            self.s = np.array(self.__h5file['/entropy'],dtype='float64')
        if '/gamma' in self.__h5file:
            self.gamma = np.array(self.__h5file['/gamma'],dtype='float64')

        if '/mu_e' in self.__h5file:
            self.mu_e = np.array(self.__h5file['/mu_e'],dtype='float64')
        if '/mu_n' in self.__h5file:
            self.mu_n = np.array(self.__h5file['/mu_n'],dtype='float64')
        if '/mu_p' in self.__h5file:
            self.mu_p = np.array(self.__h5file['/mu_p'],dtype='float64')

        if '/mu_nu' in self.__h5file:
            self.mu_nu = np.array(self.__h5file['/mu_nu'],dtype='float64')
        elif '/munu' in self.__h5file:
            self.mu_nu = np.array(self.__h5file['/munu'],dtype='float64')

        if '/Abar' in self.__h5file:
            self.abar = np.array(self.__h5file['/Abar'],dtype='float64')
        if '/Zbar' in self.__h5file:
            self.zbar = np.array(self.__h5file['/Zbar'],dtype='float64')

        if '/Xn' in self.__h5file:
            self.xn = np.array(self.__h5file['/Xn'],dtype='float64')
        if '/Xp' in self.__h5file:
            self.xp = np.array(self.__h5file['/Xp'],dtype='float64')
        if '/Xa' in self.__h5file:
            self.xa = np.array(self.__h5file['/Xa'],dtype='float64')
        if '/Xh' in self.__h5file:
            self.xh = np.array(self.__h5file['/Xh'],dtype='float64')

        if '/qvol' in self.__h5file:
            self.qvol = np.array(self.__h5file['/qvol'],dtype='float64')

        if '/yp' in self.__h5file:
            self.yp = np.array(self.__h5file['/yp'])
        if '/ypi' in self.__h5file:
            self.ypi = np.array(self.__h5file['/ypi'])
        if '/ypi0' in self.__h5file:
            self.ypi0 = np.array(self.__h5file['/ypi0'])
        if '/ypic' in self.__h5file:
            self.ypic = np.array(self.__h5file['/ypic'])
        if '/ppi' in self.__h5file:
            self.ppi = np.array(self.__h5file['/ppi'])
            self.ppi=10.0**self.ppi[:]
        if '/ppi0' in self.__h5file:
            self.ppi0 = np.array(self.__h5file['/ppi0'])
            self.ppi0=10.0**self.ppi0[:]
        if '/upi' in self.__h5file:
            self.upi = np.array(self.__h5file['/upi'])
            self.upi=10.0**self.upi[:]
        if '/upi0' in self.__h5file:
            self.upi0 = np.array(self.__h5file['/upi0'])
            self.upi0=10.0**self.upi0[:]
        if '/spi' in self.__h5file:
            self.spi = np.array(self.__h5file['/spi'])
        if '/mu_pi' in self.__h5file:
            self.mu_pi = np.array(self.__h5file['/mu_pi'])
        if '/conv' in self.__h5file:
            self.conv = np.array(self.__h5file['/conv'])
        self.__h5file.close()

    def __readtable__(self,yidx,tidx,ridx):
        result=np.empty([26],dtype='float64')
        result[0] = self.p[yidx,tidx,ridx] 
        result[1] = self.eps[yidx,tidx,ridx] 
        result[2] = self.cs2[yidx,tidx,ridx] 
        result[3] = self.s[yidx,tidx,ridx] 
        result[4] = self.gamma[yidx,tidx,ridx] 
        result[5] = self.mu_e[yidx,tidx,ridx] 
        result[6] = self.mu_n[yidx,tidx,ridx] 
        result[7] = self.mu_p[yidx,tidx,ridx] 
        result[8] = self.mu_nu[yidx,tidx,ridx] 
        result[9] = self.abar[yidx,tidx,ridx] 
        result[10] = self.zbar[yidx,tidx,ridx] 
        result[11] = self.xn[yidx,tidx,ridx] 
        result[12] = self.xp[yidx,tidx,ridx] 
        result[13] = self.xa[yidx,tidx,ridx] 
        result[14] = self.xh[yidx,tidx,ridx] 
        if type(self.qvol) != type(None):
          result[15] = self.qvol[yidx,tidx,ridx]
        if type(self.yp) != type(None):
          result[15] = self.yp[yidx,tidx,ridx]
          result[16] = self.ypi[yidx,tidx,ridx]
          result[17] = self.ypi0[yidx,tidx,ridx]
          result[18] = self.ypic[yidx,tidx,ridx]
          result[19] = self.ppi[yidx,tidx,ridx]
          result[20] = self.ppi0[yidx,tidx,ridx]
          result[21] = self.upi[yidx,tidx,ridx]
          result[22] = self.upi0[yidx,tidx,ridx]
          result[23] = self.spi[yidx,tidx,ridx]
          result[24] = self.mu_pi[yidx,tidx,ridx]
          result[25] = self.conv[yidx,tidx,ridx]
        return result

    def __interpolate__(self,yidx,tidx,ridx,yq,tq,rq):
        result=np.zeros((len(yidx),26),dtype='float64')
        for i in range(0,ridx.shape[0]):
           var1 = self.__readtable__(yidx[i],  tidx[i],  ridx[i])
           var2 = self.__readtable__(yidx[i]+1,tidx[i],  ridx[i])
           var3 = self.__readtable__(yidx[i],  tidx[i]+1,ridx[i])
           var4 = self.__readtable__(yidx[i]+1,tidx[i]+1,ridx[i])
           var5 = self.__readtable__(yidx[i],  tidx[i],  ridx[i]+1)
           var6 = self.__readtable__(yidx[i]+1,tidx[i],  ridx[i]+1)
           var7 = self.__readtable__(yidx[i],  tidx[i]+1,ridx[i]+1)
           var8 = self.__readtable__(yidx[i]+1,tidx[i]+1,ridx[i]+1)

           result[i,:] = var1*(1.0-yq[i])*(1.0-tq[i])*(1.0-rq[i]) \
                         +var2*yq[i]*(1.0-tq[i])*(1.0-rq[i]) \
                         +var3*(1.0-yq[i])*tq[i]*(1.0-rq[i]) \
                         +var4*yq[i]*tq[i]*(1.0-rq[i]) \
                         +var5*(1.0-yq[i])*(1.0-tq[i])*rq[i] \
                         +var6*yq[i]*(1.0-tq[i])*rq[i] \
                         +var7*(1.0-yq[i])*tq[i]*rq[i] \
                         +var8*yq[i]*tq[i]*rq[i]
        return result

    def print_order(self):
        print(0,'p',1,'eps',2,'cs2',3,'s',4,'gamma',5,'mu_e',6,'mu_n',7,'mu_p',8,'mu_nu',9,'abar',10,'zbar',11,'xn',12,'xp',13,'xa',14,'xh', \
                15,'qvol/yp',16,'ypi',17,'ypi0',18,'ypic',19,'ppi',20,'ppi0',21,'upi',22,'upi0',23,'spi',24,'mu_pi',24,'conv')

    def readeos(self,rho,ye,temp):
        """
        read eos from density,electron fraction and temperature
        Args:
            @rho (float): 1d array of density [g/cm3]
            @ye (float): 1d array of electron fraction
            @temp (float): 1d array of temperature [MeV]
        """
        ridx=self.rho.searchsorted(rho,side='right')-1
        yidx=self.ye.searchsorted(ye,side='right')-1
        tidx=self.temp.searchsorted(temp,side='right')-1
        ridx[ridx > self.rho.size-2] = self.rho.size-2
        yidx[yidx > self.ye.size-2] = self.ye.size-2
        tidx[tidx > self.temp.size-2] = self.temp.size-2
        ridx[ridx < 0] = 0
        yidx[yidx < 0] = 0
        tidx[tidx < 0] = 0
        rq=(rho-self.rho[ridx])/(self.rho[ridx+1]-self.rho[ridx])
        yq=(ye-self.ye[yidx])/(self.ye[yidx+1]-self.ye[yidx])
        tq=(temp-self.temp[tidx])/(self.temp[tidx+1]-self.temp[tidx])
        return self.__interpolate__(yidx,tidx,ridx,yq,tq,rq)

    def find_temperature(self,rho,ye,var,s=None,p=None,mpi=None):
        """
        find temperature from density,electron fraction and internal energy
        Args:
            @rho (float): 1d array of density [g/cm3]
            @ye (float): 1d array of electron fraction
            @var (float): 1d array of internal energy (default) [erg/g] or entropy [kb/baryon] or pressure [erg/cm3]
            @s (int): Non-None = var is entropy
            @p (int): Non-None = var is pressure
        """
        ridx=self.rho.searchsorted(rho,side='right')-1
        yidx=self.ye.searchsorted(ye,side='right')-1
        ridx[ridx > self.rho.size-2] = self.rho.size-2
        yidx[yidx > self.ye.size-2] = self.ye.size-2
        ridx[ridx < 0] = 0
        yidx[yidx < 0] = 0
        rq=(rho-self.rho[ridx])/(self.rho[ridx+1]-self.rho[ridx])
        yq=(ye-self.ye[yidx])/(self.ye[yidx+1]-self.ye[yidx])

        var_eos=np.zeros((len(self.temp)),dtype='float64')
        temp_eos=np.zeros((len(var)),dtype='float64')

        if s != None:
          vartmp = self.s
          if mpi !=None:
              vartmp[:, :, 220:] = vartmp[:, :, 220:] + self.spi[:, :, 220:]
        elif p != None:
          vartmp = self.p
          if mpi !=None:
              vartmp[:, :, 220:] = vartmp[:, :, 220:] + self.ppi[:, :, 220:] + self.ppi0[:, :, 220:]
        else:
          vartmp = self.eps
          if mpi !=None:
              vartmp[:, :, 220:] = vartmp[:, :, 220:] + self.upi[:, :, 220:] + self.upi0[:, :, 220:] + mpi * self.conv_MeVtoErg * self.ypic[:, :, 220:] / self.mb

        for i in range(0,ridx.shape[0]):
            var_eos=vartmp[yidx[i],:,ridx[i]]*(1.0-rq[i])*(1.0-yq[i]) \
                +vartmp[yidx[i],:,ridx[i]+1]*(1.0-rq[i])*yq[i] \
                +vartmp[yidx[i],:,ridx[i]+1]*rq[i]*(1.0-yq[i]) \
                +vartmp[yidx[i]+1,:,ridx[i]+1]*rq[i]*yq[i]
            tidx = np.abs(var_eos - var[i]).argmin()# - 1
            if tidx > self.temp.size-2:
                tidx = self.temp.size-2
            if tidx < 0:
                tidx = 0
            varq=(var[i]-var_eos[tidx])/(var_eos[tidx+1]-var_eos[tidx])
            temp_eos[i]=self.temp[tidx]*(1.0-varq)+self.temp[tidx+1]*varq
            temp_eos[temp_eos < self.temp[0]]  = self.temp[0]
            temp_eos[temp_eos > self.temp[-1]] = self.temp[-1]

        return temp_eos

    def find_density(self,temp,ye,var,pion=None):
        """
        find temperature from density,electron fraction and internal energy
        Args:
            @rho (float): 1d array of density [g/cm3]
            @ye (float): 1d array of electron fraction
            @var (float): 1d array of pressure [erg/cm3]
        """
          
        tidx=self.temp.searchsorted(temp,side='right')-1
        yidx=self.ye.searchsorted(ye,side='right')-1
        tq=(temp-self.temp[tidx])/(self.temp[tidx+1]-self.temp[tidx])
        yq=(ye-self.ye[yidx])/(self.ye[yidx+1]-self.ye[yidx])

        var_eos=np.zeros((len(self.rho)),dtype='float64')
        rho_eos=np.zeros((len(var)),dtype='float64')

        if pion == None:
            vartmp = self.p
        else:
            vartmp = self.p + self.ppi + self.ppi0

        for i in range(0,tidx.shape[0]):
            var_eos=vartmp[yidx[i],tidx[i],:]*(1.0-tq[i])*(1.0-yq[i]) \
                +vartmp[yidx[i],tidx[i]+1,:]*(1.0-tq[i])*yq[i] \
                +vartmp[yidx[i],tidx[i]+1,:]*tq[i]*(1.0-yq[i]) \
                +vartmp[yidx[i]+1,tidx[i]+1,:]*tq[i]*yq[i]
            ridx=var_eos.searchsorted(var[i],side='right')-1
            varq=(var[i]-var_eos[ridx])/(var_eos[ridx+1]-var_eos[ridx])
            rho_eos[i]=self.rho[ridx]*(1.0-varq)+self.rho[ridx+1]*varq

        return rho_eos

    def func(self, rho, temp, ye, p, s, pion=None):
        #print(rho,temp,ye)
        result = self.readeos(rho=rho, temp=temp, ye=ye)
        f = np.zeros((rho.shape[0], 2), dtype='float64')
        if pion == None:
            f[:, 0] = result[:, 0] - p
            f[:, 1] = result[:, 3] - s
        else:
            f[:, 0] = (result[:, 0] + result[:, 19] + result[:, 20]) - p
            f[:, 1] = (result[:, 3] + result[:, 23])  - s
        return f

    def dfunc(self, rho, temp, ye, pion=None):

        h = 1e-6
        result0 = self.readeos(rho=rho, temp=temp, ye=ye)
        result1 = self.readeos(rho=rho * (1.0 + h), temp=temp, ye=ye)
        result2 = self.readeos(rho=rho, temp=temp * (1.0 + h), ye=ye)

        J = np.zeros((rho.shape[0], 2, 2), dtype='float64')

        if pion == None:
            # drhodp
            J[:, 0, 0] = (result1[:, 0] - result0[:, 0]) / (rho * h)
            # dtempdp
            J[:, 0, 1] = (result2[:, 0] - result0[:, 0]) / (temp * h)
            # drhods
            J[:, 1, 0] = (result1[:, 3] - result0[:, 3]) / (rho * h)
            #dtempds
            J[:, 1, 1] = (result2[:, 3] - result0[:, 3]) / (temp * h)
        else:
            # drhodp
            J[:, 0, 0] = ((result1[:, 0]+result1[:, 19]+result1[:, 20]) - (result0[:, 0]+result0[:, 19]+result0[:, 20])) / (rho * h)
            # dtempdp
            J[:, 0, 1] = ((result2[:, 0]+result2[:, 19]+result2[:, 20]) - (result0[:, 0]+result0[:, 19]+result0[:, 20])) / (temp * h)
            # drhods
            J[:, 1, 0] = ((result1[:, 3]+result1[:, 23]) - (result0[:, 3]+result0[:, 23])) / (rho * h)
            #dtempds
            J[:, 1, 1] = ((result2[:, 3]+result2[:, 23]) - (result0[:, 3]+result0[:, 23])) / (temp * h)

        return J
 
    def find_density_temperature(self, rho, temp, ye, p, s, maxiter, maxerr, pion=None):
   
        df = np.zeros((rho.shape[0], 2), dtype='float64')
    
        for i in range(0, maxiter):
            f = self.func(rho=rho, temp=temp, ye=ye, p=p, s=s, pion=pion)
            J = self.dfunc(rho=rho, temp=temp, ye=ye, pion=pion)
            Jinv = np.linalg.inv(J)
    
            df[:, 0] = -(Jinv[:, 0, 0] * f[:, 0] + Jinv[:, 0, 1] * f[:, 1])
            df[:, 1] = -(Jinv[:, 1, 0] * f[:, 0] + Jinv[:, 1, 1] * f[:, 1])

            rho = rho + df[:, 0]
            rho[rho > self.rho[-1]] = self.rho[-1]
            rho[rho < self.rho[0]] = self.rho[0]

            temp = temp + df[:, 1]
            temp[temp > self.temp[-1]] = self.temp[-1]
            temp[temp < self.temp[0]] = self.temp[0]
    
            error = max(max(np.abs(df[:, 0]) / rho), max(np.abs(df[:, 1]) / temp))
            if error < maxerr:
                break
    
        return rho,temp

class eos_muon():
    """
    class for equation of state 
    Attributes:
        filepath (str) : path to eos file
        ntemp (int): number of data points in temperature
        nmu (int): number of data points in muon chemical potential
        temp (double): 1d array for temperature values [MeV]
        mu (double): 1d array for chemical potential values [MeV]
        rho (double): 3d array for density values [1/cm3]
        p (double) :: 3d array for pressure values [erg/cm3]
        eps (double) :: 3d array for internal energy values [erg/gm]
        s (double) :: 3d array for entropy values [kb/lepton]
    """

    def __init__(self):
        super().__init__()
        self.filename = None
        self.ntemp = None
        self.nmu = None
        
        self.temp = None
        self.mu = None
        self.rho = None
        self.p = None
        self.eps = None
        self.s = None
        
        self.m_mu_MeV = 105.66 
        self.conv_MeVtoErg = 1.602e-6
        self.mb = 1.66e-24
        
    def readeostable(self, filename, ntemp, nmu):
        
        self.filename = filename
        self.ntemp = ntemp
        self.nmu = nmu
        
        temp_eos = np.zeros((ntemp))
        table = []
        
        f = open(filename, 'r')
        data = f.readlines()
        f.close()
        
        for x in data:
            if x.find(' \n') == -1:
                string_list = x.split()
                if string_list:
                    table.append(string_list)        
        
        eostable = np.array(table, dtype='float64')
        eostable = eostable.reshape((self.ntemp, self.nmu, 6))
        
        self.temp = eostable[:, 0, 0]
        self.mu = eostable[0, :, -1]
        self.rho = eostable[:, :, 1] * 1e39 # 1/cm3
        self.p = eostable[:, :, 2] * self.conv_MeVtoErg * 1e39 # erg/cm3
        self.eps = eostable[:, :, 3] * self.conv_MeVtoErg * 1e39 # erg/cm3
        self.s = eostable[:, :, 4]
        
    def __readtable__(self, tidx, muidx):
        result=np.empty([4], dtype='float64')
        result[0] = self.rho[tidx, muidx]
        result[1] = self.p[tidx, muidx]
        result[2] = self.eps[tidx, muidx]
        result[3] = self.s[tidx, muidx]
        return result

    def __interpolate__(self, tidx, muidx, tq, muq):
        result=np.zeros((len(tidx), 4), dtype='float64')
        for i in range(0, tidx.shape[0]):
            var1 = self.__readtable__(tidx[i],   muidx[i])
            var2 = self.__readtable__(tidx[i]+1, muidx[i])
            var3 = self.__readtable__(tidx[i],   muidx[i]+1)
            var4 = self.__readtable__(tidx[i]+1, muidx[i]+1)

            result[i,:] = var1 * (1.0 - tq[i]) * (1.0 - muq[i]) \
                         + var2 * tq[i] * (1.0 - muq[i]) \
                         + var3 * (1.0 - tq[i]) * muq[i] \
                         + var4 * tq[i] * muq[i]
        return result

    def print_order(self):
        print(0, 'rho [1/cm^3],', 1, 'p [erg/cm^3],', 2, 'eps [erg/g],', 3, 's [kb/lepton]')
        
    def readeos(self, temp, mu):
        
        tidx = self.temp.searchsorted(temp, side='right') - 1
        muidx = self.mu.searchsorted(mu, side='right') - 1
        
        tidx[tidx > self.temp.size-2] = self.temp.size-2
        muidx[muidx > self.mu.size-2] = self.mu.size-2
        
        tidx[tidx < 0] = 0
        muidx[muidx < 0] = 0
        
        tq = (temp - self.temp[tidx]) / (self.temp[tidx+1] - self.temp[tidx])
        muq = (mu - self.mu[muidx]) / (self.mu[muidx+1] - self.mu[muidx])
        
        return self.__interpolate__(tidx, muidx, tq, muq)
    
    def find_mu(self, temp, var):

        tidx = self.temp.searchsorted(temp, side='right') - 1
        tidx[tidx > self.temp.size-2] = self.temp.size-2
        tidx[tidx < 0] = 0
        tq = (temp - self.temp[tidx]) / (self.temp[tidx+1] - self.temp[tidx])

        var_eos = np.zeros((len(self.mu)),dtype='float64')
        mu_eos = np.zeros((len(var)),dtype='float64')

        var_eos_1 = np.zeros((len(self.mu)),dtype='float64')
        var_eos_2 = np.zeros((len(self.mu)),dtype='float64')

        vartmp = self.rho

        for i in range(0, tidx.shape[0]):
            
            var_eos = vartmp[tidx[i], :] * (1.0 - tq[i]) \
                + vartmp[tidx[i]+1, :] * tq[i]
   
            muidx = np.abs(var_eos - var[i]).argmin()# - 1
            if muidx > self.mu.size-2:
                muidx = self.mu.size-2
            if muidx < 0:
                muidx = 0

            varq = (var[i] - var_eos[muidx]) / (var_eos[muidx+1] - var_eos[muidx])
            mu_eos[i] = self.mu[muidx] * (1.0 - varq) + self.mu[muidx+1] * varq

        mu_eos[mu_eos < self.mu[0]]  = self.mu[0]
        mu_eos[mu_eos > self.mu[-1]] = self.mu[-1]
        mu_eos[np.isnan(mu_eos)]  = 0.0
        mu_eos[var < 1e-36]  = 0.0

        return mu_eos
