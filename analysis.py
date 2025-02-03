'''
Created on Feb 16, 2017

@author: ninoy
'''
import numpy as np
from pynada import pynada
from util import dxdf1
from util import dydf1
from util import find_1d
from scipy.special import sph_harm as ylm
import scipy.special as sps
from eos import eos

class analysis(pynada):
    """
    class for analyzing NADA output
    """

    def __init__(self):

        super().__init__()

        self.g_grav = 6.673e-8           	# gravitational constant in cm^3 g^-1 s^-2
        self.c_light = 2.99792458e10     	# speed of light in cm s^-1
        self.k_boltzmann = 1.380658e-16  	# Boltzmann's constant in erg K^-1
        self.h_planck = 6.626176e-27     	# Planck's constant in erg s
        self.n_avogadro = 6.0221415e23   	# Avogadro's number
        self.m_u = 1.6605402e-24         	# atomic mass unit in g
        self.mb = 1.66e-24          		# baryon mass unit in g
        self.conv_MeVtoErg = 1.602e-6		# conversion factor from MeV to erg
        self.conv_ErgtoMeV = 624150.647996e0 	# conversion factor from erg to MeV
        self.rs = 1.47706e5			# half schwarzschild radius rs=G*M_solar/c**2 in cm
        self.Msolar = 1.98892e33		# solar mass in g
        self.mec2 = 0.511                       # electron mass in MeV
        self.G2_bruenn = 5.18e-44               # Weak coupling constant Bruenn (C15) MeV^-2 cm^2
        self.sigma0 = 1.76e-44                  # Weak interaction cross-section cm^2
        self.sin2thW = 0.23                     # Weinberg angle
        self.CA = 0.5
        self.CV = 0.5 + 2.0*0.23

    def neutrino_energy_density(self,ts,te):
        """
        calculate neutrino energy density
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        f = np.array(self.readvar('J',tstart=ts,tend=te),dtype='float64')
        f = f*self.de[None,None,None,None,None,:,None]
        return f*self.conv_MeVtoErg 

    def neutrino_total_neutrino_energy(self,ts,te):
        """
        calculate cell mass
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        E = self.neutrino_energy_density(ts,te).sum(axis=(5))
        return E[:,:,:,:,:,0]+E[:,:,:,:,:,1]+4.0*E[:,:,:,:,:,2]

    def neutrino_equilibrium_energy_density(self,ts,te):
        """
        calculate neutrino equilibrium energy density
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        f = np.array(self.readvar('Je',tstart=ts,tend=te),dtype='float64')
        f[:,:,:,:,:,:,:] = f[:,:,:,:,:,:,:]*self.de[None,None,None,None,None,:,None]*self.conv_MeVtoErg 
        return f

    def energy_integrate(self,varname,ts,te,p=0,flux=None,scale=None):
        """
        energy integration integral_(var*e^p*(J/e)*de)
        Args:
            @varname (str): variable name or variable
            @te (int): ending time step
            @ts (int): starting time step
            @p (float): power of energy
            @flux (Optional[int]): 0 = scale with total number flux H/e
                                   1 = scale with radial number flux H1/e
                                   2 = scale with polar number flux H2/e
                                   3 = scale with azimuthal number flux H3/e 
                                   None = scale with number density J/e
            @scale (Optional[int]): Non None = divide by integral_((J/e)*de)
        """
        self.touch(time=te)
        if flux == None:
            f = self.readvar('J',tstart=ts,tend=te)
        elif flux == 1:
            f = self.readvar('H1',tstart=ts,tend=te)
        elif flux == 2:
            f = self.readvar('H2',tstart=ts,tend=te)
        elif flux == 3:
            f = self.readvar('H3',tstart=ts,tend=te)
        elif flux == 0:
            f = self.readvar('H',tstart=ts,tend=te)
        elif flux == 4:
            f = self.readvar('Je',tstart=ts,tend=te)
  
        func = np.zeros([te-ts+1,self.numpatch[0],self.nz[0],self.ny[0],self.nx[0],self.numneu[0]])
        if scale!=None:
            scalefunc = np.zeros([te-ts+1,self.numpatch[0],self.nz[0],self.ny[0],self.nx[0],self.numneu[0]])

        if isinstance(varname,str):
            var1 = self.readvar(varname,tstart=ts,tend=te)
        else:
            var1 = varname

        for l in range(0,self.ne[0]):
            if var1.ndim > 1:
                func[:,:,:,:,:,:]=func[:,:,:,:,:,:]+var1[:,:,:,:,:,l,:]*(self.e[l]**p)*(f[:,:,:,:,:,l,:]/self.e[l])*self.de[l]
            else:
                func[:,:,:,:,:,:]=func[:,:,:,:,:,:]+var1[l]*(self.e[l]**p)*(f[:,:,:,:,:,l,:]/self.e[l])*self.de[l]
            if scale!=None:
                scalefunc[:,:,:,:,:,:]=scalefunc[:,:,:,:,:,:]+(f[:,:,:,:,:,l,:]/self.e[l])*self.de[l]

        if scale!=None:
            return func/scalefunc
        else:
            return func


    def neutrino_distribution_function(self,ts,te,eq=None):
        """
        calculate neutrino distribution function
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @eq (Optional[int]): Non None=equilibrium
        """
        self.touch(time=te)
        if eq==None:
            func = self.readvar('J',tstart=ts,tend=te)
        else:
            func = self.readvar('Je',tstart=ts,tend=te)
        for l in range(0,self.ne[0]):
            fact = 4.0*np.pi*(self.e[l]*self.conv_MeVtoErg/(self.h_planck*self.c_light))**3
            func[:,:,:,:,:,l,:]=func[:,:,:,:,:,l,:]/fact
        return func

    def calculate_degeneracy_parameter(self,ts,te,sp=None):
        """
        calculate neutrino degeneracy_parameter
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @sp (Optional[str]): None = neutrino
                                'e' = electron
                                'p' = proton
                                'n' = neutron
        """
        self.touch(time=te)
        if sp==None:
            func = self.readvar('mu_nu',tstart=ts,tend=te)
        elif sp=='e':
            func = self.readvar('mu_e',tstart=ts,tend=te)
        elif sp=='p':
            func = self.readvar('mu_p',tstart=ts,tend=te)
        elif sp=='n':
            func = self.readvar('mu_n',tstart=ts,tend=te)
        temp = self.readvar('temp',tstart=ts,tend=te)
        return func/temp

    def calculate_cell_area(self,ts,te,conf=None):
        """
        calculate cell area
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)

        phi = self.readvar('phi',tstart=ts,tend=te)
        if conf == None:
            conf = np.exp(4.0*phi)
        else:
            conf = 1.0

        gxx = conf*self.readvar('gphixx',tstart=ts,tend=te)
        gxy = conf*self.readvar('gphixy',tstart=ts,tend=te)*self.x[None,None,None,:]
        gxz = conf*self.readvar('gphixz',tstart=ts,tend=te)*(self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gyy = conf*self.readvar('gphiyy',tstart=ts,tend=te)*(self.x[None,None,None,:]*self.x[None,None,None,:])
        gyz = conf*self.readvar('gphiyz',tstart=ts,tend=te)* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gzz = conf*self.readvar('gphizz',tstart=ts,tend=te)* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]* \
              np.sin(self.y[None,None,:,None])*np.sin(self.y[None,None,:,None]))

        exx = np.ones(gxx.shape)
        exy = np.zeros(exx.shape)
        exz = np.zeros(exx.shape)
        eyx = np.zeros(exx.shape)
        eyy = np.ones(exx.shape)
        eyz = np.zeros(exx.shape)
        ezx = np.zeros(exx.shape)
        ezy = np.zeros(exx.shape)
        ezz = np.ones(exx.shape)

        g11 = gxx*exx*exx+gyy*eyx*eyx+gzz*ezx*ezx+2.0*(gxy*exx*eyx+gxz*exx*ezx+gyz*eyx*ezx)
        g22 = gxx*exy*exy+gyy*eyy*eyy+gzz*ezy*ezy+2.0*(gxy*exy*eyy+gxz*exy*ezy+gyz*eyy*ezy)
        g33 = gxx*exz*exz+gyy*eyz*eyz+gzz*ezz*ezz+2.0*(gxy*exz*eyz+gxz*exz*ezz+gyz*eyz*ezz)
        g12 = gxx*exx*exy+gyy*eyx*eyy+gzz*ezx*ezy+gxy*exx*eyy+gxy*eyx*exy+gxz*exx*ezy+gxz*ezx*exy+gyz*eyx*ezy+gyz*ezx*eyy
        g13 = gxx*exx*exz+gyy*eyx*eyz+gzz*ezx*ezz+gxy*exx*eyz+gxy*eyx*exz+gxz*exx*ezz+gxz*ezx*exz+gyz*eyx*ezz+gyz*ezx*eyz
        g23 = gxx*exy*exz+gyy*eyy*eyz+gzz*ezy*ezz+gxy*exy*eyz+gxy*eyy*exz+gxz*exy*ezz+gxz*ezy*exz+gyz*eyy*ezz+gyz*ezy*eyz

        exy = exy - exx*g12/g11
        eyy = eyy - eyx*g12/g11
        ezy = ezy - ezx*g12/g11

        exz = exz - exx*g13/g11 - exy*g23/g22
        eyz = eyz - eyx*g13/g11 - eyy*g23/g22
        ezz = ezz - ezx*g13/g11 - ezy*g23/g22

        g11 = gxx*exx*exx+gyy*eyx*eyx+gzz*ezx*ezx+2.0*(gxy*exx*eyx+gxz*exx*ezx+gyz*eyx*ezx)
        g22 = gxx*exy*exy+gyy*eyy*eyy+gzz*ezy*ezy+2.0*(gxy*exy*eyy+gxz*exy*ezy+gyz*eyy*ezy)
        g33 = gxx*exz*exz+gyy*eyz*eyz+gzz*ezz*ezz+2.0*(gxy*exz*eyz+gxz*exz*ezz+gyz*eyz*ezz)
        g12 = gxx*exx*exy+gyy*eyx*eyy+gzz*ezx*ezy+gxy*exx*eyy+gxy*eyx*exy+gxz*exx*ezy+gxz*ezx*exy+gyz*eyx*ezy+gyz*ezx*eyy
        g13 = gxx*exx*exz+gyy*eyx*eyz+gzz*ezx*ezz+gxy*exx*eyz+gxy*eyx*exz+gxz*exx*ezz+gxz*ezx*exz+gyz*eyx*ezz+gyz*ezx*eyz
        g23 = gxx*exy*exz+gyy*eyy*eyz+gzz*ezy*ezz+gxy*exy*eyz+gxy*eyy*exz+gxz*exy*ezz+gxz*ezy*exz+gyz*eyy*ezz+gyz*ezy*eyz

        exx = exx/np.sqrt(g11)
        eyx = eyx/np.sqrt(g11)
        ezx = ezx/np.sqrt(g11)

        exy = exy/np.sqrt(g22)
        eyy = eyy/np.sqrt(g22)
        ezy = ezy/np.sqrt(g22)

        exz = exz/np.sqrt(g33)
        eyz = eyz/np.sqrt(g33)
        ezz = ezz/np.sqrt(g33)

        myy = g22-eyx*eyx
        myz = g23-eyx*ezx

        mzy = g23-ezx*eyx
        mzz = g33-ezx*ezx

        sqdetm = np.sqrt(myy*mzz-myz*mzy)

        return sqdetm*self.dy[None,None,:,None]*self.dz[None,:,None,None]

    def calculate_cell_areal_radius(self,ts,te):
        """
        calculate cell areal radius
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)

        phi = self.readvar('phi',tstart=ts,tend=te)
        conf = np.exp(2.0*phi)

        return conf*self.x[None,None,None,:]

    def calculate_cell_radial_length(self,ts,te):
        """
        calculate cell radial length
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)

        phi = self.readvar('phi',tstart=ts,tend=te)
        conf = np.exp(4.0*phi)

        gxx = conf*self.readvar('gphixx',tstart=ts,tend=te)
        return np.sqrt(gxx)*self.dx[None,None,None,:]

    def calculate_radial_distance(self,ts,te):
        """
        calculate radial distance
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        dr = self.calculate_cell_radial_length(ts,te)
        return dr.cumsum(axis=(3)) 

    def calculate_radial_mass_flow_rate(self,ts,te,ray=None):
        """
        calculate mass coordinate
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        area = np.array(self.calculate_cell_area(ts,te),dtype='float64')
        rho = self.readvar('rho',tstart=ts,tend=te)
        w = self.readvar('lorentz',tstart=ts,tend=te)
        vr = self.readvar('velocity/velx',tstart=ts,tend=te)
        br = self.readvar('betau/betaux',tstart=ts,tend=te)
        alp = self.readvar('alp',tstart=ts,tend=te)
        flux = np.array(rho*w*(vr-br/alp),dtype='float64')
        mdot = flux*area*self.c_light
        if ray ==  None:
          mdot = mdot.sum(axis=(1,2))
        return mdot

    def calculate_cell_covariant_vx(self,ts,te):
        """
        calculate cell covariant vx
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        phi = self.readvar('phi',tstart=ts,tend=te)
        conf = np.exp(4.0*phi)
        gxx = conf*self.readvar('gphixx',tstart=ts,tend=te)
        gxy = conf*self.readvar('gphixy',tstart=ts,tend=te)*self.x[None,None,None,:]
        gxz = conf*self.readvar('gphixz',tstart=ts,tend=te)*(self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        velx = self.readvar('velocity/velx',tstart=ts,tend=te)
        vely = self.readvar('velocity/vely',tstart=ts,tend=te)
        velz = self.readvar('velocity/velz',tstart=ts,tend=te)
        return gxx*velx+gxy*vely+gxz*velz

    def calculate_cell_covariant_vy(self,ts,te):
        """
        calculate cell covariant vy
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        phi = self.readvar('phi',tstart=ts,tend=te)
        conf = np.exp(4.0*phi)
        gxy = conf*self.readvar('gphixy',tstart=ts,tend=te)*self.x[None,None,None,:]
        gyy = conf*self.readvar('gphiyy',tstart=ts,tend=te)*(self.x[None,None,None,:]*self.x[None,None,None,:])
        gyz = conf*self.readvar('gphiyz',tstart=ts,tend=te)* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        velx = self.readvar('velocity/velx',tstart=ts,tend=te)
        vely = self.readvar('velocity/vely',tstart=ts,tend=te)
        velz = self.readvar('velocity/velz',tstart=ts,tend=te)
        return gxy*velx+gyy*vely+gyz*velz

    def calculate_cell_covariant_vz(self,ts,te):
        """
        calculate cell covariant vz
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        phi = self.readvar('phi',tstart=ts,tend=te)
        conf = np.exp(4.0*phi)
        gxz = conf*self.readvar('gphixz',tstart=ts,tend=te)*(self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gyz = conf*self.readvar('gphiyz',tstart=ts,tend=te)* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gzz = conf*self.readvar('gphizz',tstart=ts,tend=te)* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]* \
              np.sin(self.y[None,None,:,None])*np.sin(self.y[None,None,:,None]))
        velx = self.readvar('velocity/velx',tstart=ts,tend=te)
        vely = self.readvar('velocity/vely',tstart=ts,tend=te)
        velz = self.readvar('velocity/velz',tstart=ts,tend=te)
        return gxz*velx+gyz*vely+gzz*velz

    def calculate_cell_tetrad(self,ts,te,inv=None):
        """
        calculate cell area
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)

        phi = self.readvar('phi',tstart=ts,tend=te)
        conf = np.exp(4.0*phi)

        gxx = conf*self.readvar('gphixx',tstart=ts,tend=te)
        gxy = conf*self.readvar('gphixy',tstart=ts,tend=te)*self.x[None,None,None,:]
        gxz = conf*self.readvar('gphixz',tstart=ts,tend=te)*(self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gyy = conf*self.readvar('gphiyy',tstart=ts,tend=te)*(self.x[None,None,None,:]*self.x[None,None,None,:])
        gyz = conf*self.readvar('gphiyz',tstart=ts,tend=te)* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gzz = conf*self.readvar('gphizz',tstart=ts,tend=te)* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]* \
              np.sin(self.y[None,None,:,None])*np.sin(self.y[None,None,:,None]))

        exx = np.ones(gxx.shape)
        exy = np.zeros(exx.shape)
        exz = np.zeros(exx.shape)
        eyx = np.zeros(exx.shape)
        eyy = np.ones(exx.shape)
        eyz = np.zeros(exx.shape)
        ezx = np.zeros(exx.shape)
        ezy = np.zeros(exx.shape)
        ezz = np.ones(exx.shape)

        g11 = gxx*exx*exx+gyy*eyx*eyx+gzz*ezx*ezx+2.0*(gxy*exx*eyx+gxz*exx*ezx+gyz*eyx*ezx)
        g22 = gxx*exy*exy+gyy*eyy*eyy+gzz*ezy*ezy+2.0*(gxy*exy*eyy+gxz*exy*ezy+gyz*eyy*ezy)
        g33 = gxx*exz*exz+gyy*eyz*eyz+gzz*ezz*ezz+2.0*(gxy*exz*eyz+gxz*exz*ezz+gyz*eyz*ezz)
        g12 = gxx*exx*exy+gyy*eyx*eyy+gzz*ezx*ezy+gxy*exx*eyy+gxy*eyx*exy+gxz*exx*ezy+gxz*ezx*exy+gyz*eyx*ezy+gyz*ezx*eyy
        g13 = gxx*exx*exz+gyy*eyx*eyz+gzz*ezx*ezz+gxy*exx*eyz+gxy*eyx*exz+gxz*exx*ezz+gxz*ezx*exz+gyz*eyx*ezz+gyz*ezx*eyz
        g23 = gxx*exy*exz+gyy*eyy*eyz+gzz*ezy*ezz+gxy*exy*eyz+gxy*eyy*exz+gxz*exy*ezz+gxz*ezy*exz+gyz*eyy*ezz+gyz*ezy*eyz

        exy = exy - exx*g12/g11
        eyy = eyy - eyx*g12/g11
        ezy = ezy - ezx*g12/g11

        exz = exz - exx*g13/g11 - exy*g23/g22
        eyz = eyz - eyx*g13/g11 - eyy*g23/g22
        ezz = ezz - ezx*g13/g11 - ezy*g23/g22

        g11 = gxx*exx*exx+gyy*eyx*eyx+gzz*ezx*ezx+2.0*(gxy*exx*eyx+gxz*exx*ezx+gyz*eyx*ezx)
        g22 = gxx*exy*exy+gyy*eyy*eyy+gzz*ezy*ezy+2.0*(gxy*exy*eyy+gxz*exy*ezy+gyz*eyy*ezy)
        g33 = gxx*exz*exz+gyy*eyz*eyz+gzz*ezz*ezz+2.0*(gxy*exz*eyz+gxz*exz*ezz+gyz*eyz*ezz)
        g12 = gxx*exx*exy+gyy*eyx*eyy+gzz*ezx*ezy+gxy*exx*eyy+gxy*eyx*exy+gxz*exx*ezy+gxz*ezx*exy+gyz*eyx*ezy+gyz*ezx*eyy
        g13 = gxx*exx*exz+gyy*eyx*eyz+gzz*ezx*ezz+gxy*exx*eyz+gxy*eyx*exz+gxz*exx*ezz+gxz*ezx*exz+gyz*eyx*ezz+gyz*ezx*eyz
        g23 = gxx*exy*exz+gyy*eyy*eyz+gzz*ezy*ezz+gxy*exy*eyz+gxy*eyy*exz+gxz*exy*ezz+gxz*ezy*exz+gyz*eyy*ezz+gyz*ezy*eyz

        exx = exx/np.sqrt(g11)
        eyx = eyx/np.sqrt(g11)
        ezx = ezx/np.sqrt(g11)

        exy = exy/np.sqrt(g22)
        eyy = eyy/np.sqrt(g22)
        ezy = ezy/np.sqrt(g22)

        exz = exz/np.sqrt(g33)
        eyz = eyz/np.sqrt(g33)
        ezz = ezz/np.sqrt(g33)

        if inv !=None:
          iexx = g11*exx + g12*eyx + g13*ezx
          iexy = g11*exy + g12*eyy + g13*ezy
          iexz = g11*exz + g12*eyz + g13*ezz
  
          ieyx = g12*exx + g22*eyx + g23*ezx
          ieyy = g12*exy + g22*eyy + g23*ezy
          ieyz = g12*exz + g22*eyz + g23*ezz
  
          iezx = g13*exx + g23*eyx + g33*ezx
          iezy = g13*exy + g23*eyy + g33*ezy
          iezz = g13*exz + g23*eyz + g33*ezz
  
          return iexx, iexy, iexz, ieyx, ieyy, ieyz, iezx, iezy, iezz 
        else: 
          return exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz 

    def calculate_cell_ortho_tetrad_velocity(self,ts,te):
        """
        calculate cell area
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)

        iexx, iexy, iexz, ieyx, ieyy, ieyz, iezx, iezy, iezz = self.calculate_cell_tetrad(ts, te, inv=1)
        vx = self.readvar('velocity/velx',tstart=ts,tend=te)
        vy = self.readvar('velocity/vely',tstart=ts,tend=te)
        vz = self.readvar('velocity/velz',tstart=ts,tend=te)

        return iexx*vx + ieyx*vy + iezx*vz, iexy*vx + ieyy*vy + iezy*vz, iexz*vx + ieyz*vy + iezz*vz

    def calculate_sqdetg(self,ts,te,conf=None,orth=None):
        """
        calculate cell sqdetg
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @conf(Optional[int]): None=Conformal
                                  Non-None=Non-conformal
            @orth(Optional[int]): None=Orthonormal
                                  Non-None=Non-orthonormal
        """
        self.touch(time=te)
        gxx = self.readvar('gphixx',tstart=ts,tend=te)
        gxy = self.readvar('gphixy',tstart=ts,tend=te)
        gxz = self.readvar('gphixz',tstart=ts,tend=te)
        gyy = self.readvar('gphiyy',tstart=ts,tend=te)
        gyz = self.readvar('gphiyz',tstart=ts,tend=te)
        gzz = self.readvar('gphizz',tstart=ts,tend=te)

        sqdetg = np.sqrt((gxx*gyy*gzz - gxx*gyz**2 - gxy**2*gzz + 2*gxy*gxz*gyz -gxz**2*gyy))

        if conf != None:
            phi = self.readvar('phi',tstart=ts,tend=te)
            conf = np.exp(6.0*phi)
            sqdetg = conf*sqdetg

        if orth != None:
            sqdetg = self.x[None,None,None,:]**2*np.sin(self.y[None,None,:,None])*sqdetg

        return sqdetg

    def calculate_cell_volume(self,ts,te,local=None):
        """
        calculate cell volume
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @local(Optional[int]): Non None=local volume
        """
        self.touch(time=te)
        phi = self.readvar('phi',tstart=ts,tend=te)

        x_r=self.dx.cumsum()
        x_l=x_r-self.dx

        y_r = self.y + 0.5*self.dy
        y_l = y_r - self.dy
        #y_r=self.dy.cumsum()
        #y_l=y_r-self.dy

        sqdetg = np.exp(6.0*phi) * self.calculate_sqdetg(ts, te)
        if sqdetg.ndim == 4:
            vol = sqdetg[:,:,:,:]*\
                  (1.0/3.0)*(x_r[None,None,None,:]**3-x_l[None,None,None,:]**3)*\
                  (np.cos(y_l[None,None,:,None])-np.cos(y_r[None,None,:,None]))*self.dz[None,:,None,None]
        elif sqdetg.ndim == 5:
            vol = sqdetg[:,:,:,:,:]*\
                  (1.0/3.0)*(x_r[None,None,None,None,:]**3-x_l[None,None,None,None,:]**3)*\
                  (np.cos(y_l[None,None,None,:,None])-np.cos(y_r[None,None,None,:,None]))*\
                  self.dz[None,None,:,None,None]

        if local != None:
            vol = vol/sqdetg
        return vol

    def calculate_cell_angular_momentum(self,ts,te,specific=None):
        """
        calculate angular momentum
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @specific (Optional[int]): if non-nan = return specific angular momentum
        """
        self.touch(time=te)
        mass = self.calculate_cell_mass(ts,te)
        w = self.readvar('lorentz',tstart=ts,tend=te)
        h = self.readvar('h',tstart=ts,tend=te)/self.c_light**2
        v1 = self.readvar('velocity/velx',tstart=ts,tend=te)
        v2 = self.readvar('velocity/vely',tstart=ts,tend=te)
        v3 = self.readvar('velocity/velz',tstart=ts,tend=te)

        alp = self.readvar('hydro/alp',tstart=ts,tend=te)
        b1 = self.readvar('hydro/betaux',tstart=ts,tend=te)
        b2 = self.readvar('hydro/betauy',tstart=ts,tend=te) / self.x[None,None,None,:]
        b3 = self.readvar('hydro/betauz',tstart=ts,tend=te) / (self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gxz = self.readvar('hydro/gxz',tstart=ts,tend=te)*\
              (self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gyz = self.readvar('hydro/gyz',tstart=ts,tend=te)*\
              (self.x[None,None,None,:]**2*np.sin(self.y[None,None,:,None]))
        gzz = self.readvar('hydro/gzz',tstart=ts,tend=te)*\
              (self.x[None,None,None,:]**2*np.sin(self.y[None,None,:,None])**2)

        vz = (gxz*(v1-b1/alp) + gyz*(v2-b2/alp) + gzz*(v3-b3/alp))*self.c_light
        if specific == None:
            return np.array(w**2*h*vz*mass,dtype='float64')
        else:
            return np.array(w**2*h*vz,dtype='float64')

    def calculate_cell_mass(self,ts,te,grav=None):
        """
        calculate cell mass
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @grav(Optional[int]): gravitational mass 
        """
        self.touch(time=te)
        rho = self.readvar('rho',tstart=ts,tend=te)
        w = self.readvar('lorentz',tstart=ts,tend=te)
        if grav == None:
            vol = np.array(self.calculate_cell_volume(ts,te),dtype='float64')
            rhow = np.array(rho*w,dtype='float64')
        else:
            vol = np.array(self.calculate_cell_volume(ts,te,local=1),dtype='float64')
            eps = self.readvar('eps',tstart=ts,tend=te)
            rhow = np.array(rho*(1.0+eps/self.c_light**2),dtype='float64')
        return rhow*vol

    def calculate_cell_lateral_kinetic_energy(self,ts,te):
        """
        calculate cell lateral kinetic energy
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        mass = self.calculate_cell_mass(ts,te)
        vra  = self.calculate_mass_average_quantity('velocity/velx',ts,te)*self.c_light 
        vaa  = self.calculate_mass_average_quantity('velocity/velz',ts,te)*self.c_light*(np.pi/4.0)*self.x[None,:]
        vr = self.readvar('velocity/velx',ts,te)*self.c_light - vra[:,None,None,:]
        vp = self.readvar('velocity/vely',ts,te)*self.x[None,None,None,:]*self.c_light
        va = self.readvar('velocity/velz',ts,te)*self.x[None,None,None,:]*np.sin(self.y[None,None,:,None])*self.c_light - vaa[:,None,None,:]
        return mass*(vr*vr+vp*vp+va*va)

    def calculate_shell_mass(self,ts,te,grav=None):
        """
        calculate cell mass
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        mass = self.calculate_cell_mass(ts,te,grav)
        ms = mass.sum(axis=(1,2))
         
        return ms

    def calculate_mass_coordinate(self,ts,te,grav=None):
        """
        calculate mass coordinate
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        ms = self.calculate_shell_mass(ts,te,grav)
        mc = ms.cumsum(axis=1)
        return mc

    def calculate_vol_average_quantity(self,var,ts,te):
        """
        calculate volume average quantities
        Args:
            @var (str): variable
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        q = self.readvar(var,tstart=ts,tend=te)
        mass = np.array(self.calculate_cell_volume(ts,te),dtype='float64')
        qm = np.array(q*mass,dtype='float64')
        return qm.sum(axis=(1,2))/mass.sum(axis=(1,2))

    def calculate_mass_average_quantity(self,var,ts,te):
        """
        calculate mass average quantities
        Args:
            @var (str): variable
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        q = self.readvar(var,tstart=ts,tend=te)
        mass = np.array(self.calculate_cell_mass(ts,te),dtype='float64')
        qm = np.array(q*mass,dtype='float64')
        return qm.sum(axis=(1,2))/mass.sum(axis=(1,2))

    def gettime(self,ts,te):
        """
        return array with time value
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        tmp=[]
        for t in range(ts,te+1):
            super().__readh5__(varname=None,time=t)
            tmp.append(self.time)
        time=np.array(tmp,dtype='float64')
        return time[:,0]

    def calculate_brunt_vaisala_frequency(self,ts,te,rot=None):
        """
        calculate brunt vaisala frequency
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        n = self.x.size
        alp = self.readvar('alp',tstart=ts,tend=te)
        phi = np.exp(self.readvar('phi',tstart=ts,tend=te))
        rho = self.readvar('rho',tstart=ts,tend=te)
        eps = self.readvar('eps',tstart=ts,tend=te)
        p = self.readvar('p',tstart=ts,tend=te)
        rhoh = rho + (rho*eps + p)/self.c_light**2
        cs2 = self.readvar('cs2',tstart=ts,tend=te)
        dalp = dxdf1(alp,self.x)
        dp = dxdf1(p,self.x)
        drhoeps = dxdf1(rho*(self.c_light**2+eps),self.x)
        factor1 = alp / (rhoh*phi**4)
        factor2 = dalp*(dp/cs2[:,:,:,1:-1]-drhoeps)
        if rot != None:
            vz2 = (self.readvar('velocity/velz',tstart=ts,tend=te)*self.c_light)**2
            dxdvz2 = dxdf1(vz2,self.x)
            dydvz2 = dydf1(vz2,self.y)
            C_s = self.x[None,None,None,1:-1]*dxdvz2[:,:,1:-1,:] + np.tan(self.y[None,None,1:-1,None])*dydvz2[:,:,:,1:-1] + 8.0*vz2[:,:,1:-1,1:-1]
            factor3 = rhoh[:,:,1:-1,1:-1]*C_s
            N2 = factor1[:,:,1:-1,1:-1]*(factor2[:,:,1:-1,:]+factor3)
        else:
            N2 = factor1[:,:,:,1:-1]*factor2
        N2[N2 > 0.0] = 0.0
        return np.sqrt(-N2)

    def calculate_chi_parameter(self,ts,te):
        """
        calculate chi parameter
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        n = self.x.size
        wBV = (self.calculate_brunt_vaisala_frequency(ts,te))**2
        vr = np.abs(self.readvar('velocity/velx',tstart=ts,tend=te))*self.c_light
        mass = np.array(self.calculate_cell_volume(ts,te),dtype='float64')
        wBVm = np.array(wBV*mass[:,:,:,1:-1],dtype='float64')
        wBVm = wBVm.sum(axis=(1,2))/mass[:,:,:,1:-1].sum(axis=(1,2))
        wBVm = np.sqrt(wBVm)
        vrm = np.array(vr*mass,dtype='float64')
        vrm = vrm.sum(axis=(1,2))/mass.sum(axis=(1,2))
        return wBVm*self.dx[None,1:-1]/vrm[:,1:-1]

    def calculate_advection_time_scale(self,ts,te):
        """
        calculate advection time scale
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        m = self.calculate_shell_mass(ts,te)
        mdot = self.calculate_radial_mass_flow_rate(ts,te)
        return m/np.abs(mdot)

    def calculate_total_energy(self,ts,te):
        """
        calculate total energy
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        alp = self.readvar('alp',tstart=ts,tend=te)
        rho = self.readvar('rho',tstart=ts,tend=te)
        w = self.readvar('lorentz',tstart=ts,tend=te)
        h = self.readvar('h',tstart=ts,tend=te)
        p = self.readvar('p',tstart=ts,tend=te)
        D = w*rho
        Dc2 = D*self.c_light**2 
        tau = w**2*rho*h-p-Dc2
        phiout = self.calculate_newtonian_potential(ts, te, phi_out=1)
        phiout = 0.5*(phiout[:, 1:] + phiout[:, :-1])
        return (alp*(tau+Dc2)-Dc2)/(D) + phiout[:,None,None,:]

    def calculate_escape_velocity(self,ts,te):
        """
        calculate total energy
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        alp = self.readvar('alp',tstart=ts,tend=te)
        return np.sqrt(1.0-alp*alp)

    def realsphericalharmonics(m,l,phi,theta):
        if m==0:
           return ylm(m,l,phi,theta)
        elif m<0:
           return np.real(np.complex(0,1)*np.sqrt(1.0/2.0)*(ylm(m,l,phi,theta)-(-1)**m*ylm(-m,l,phi,theta)))
        elif m>0:
           return np.real(np.complex(1,0)*np.sqrt(1.0/2.0)*(ylm(-m,l,phi,theta)+(-1)**m*ylm(m,l,phi,theta)))

    def spectral_decomposition_2d(self,var,n,t=None):
        """
        spectral decomposition in 2d
        Args:
            @var (float): variable to decompose
            @n (int): max multipole order
        """
        self.touch(time=t)
        l=np.arange(0,n+1)
        Y=np.real(ylm(0,l[:,None,None],self.z[None,:,None],self.y[None,None,:]))
        dOmg=np.sin(self.y[None,:])*self.dy[None,:]*self.dz[:,None]
        YdOmg=Y*dOmg[None,:,:]
        intYdOmg=np.sqrt(4.0*np.pi/(2.0*l+1)) #YdOmg.sum(axis=(1,2))
        if var.ndim == 4:
           return (var[None,:,:,:,:]*YdOmg[:,None,:,:,None]).sum(axis=(2,3))/intYdOmg[:,None,None]
        elif var.ndim == 3:
           return (var[None,:,:,:]*YdOmg[:,None,:,:]).sum(axis=(2,3))/intYdOmg[:,None]

    def spectral_decomposition_3d(self,var,n,t=None):
        """
        spectral decomposition in 3d
        Args:
            @var (float): variable to decompose
            @n (int): max multipole order
        """
        self.touch(time=t)
        l=np.arange(0,n+1)
        m=np.arange(0,n+1)
        Y=np.real(ylm(m[None,:,None,None],l[:,None,None,None],self.z[None,None,:,None],self.y[None,None,None,:]))
        Y_i=np.imag(ylm(m[None,:,None,None],l[:,None,None,None],self.z[None,None,:,None],self.y[None,None,None,:]))
        dOmg=np.sin(self.y[None,:])*self.dy[None,:]*self.dz[:,None]
        YdOmg=Y*dOmg[None,None,:,:]
        intYdOmg=np.sqrt(2.0*np.pi/(2.0*l[:,None]+1))*np.sqrt(sps.factorial(l[:,None]+m[None,:])/np.maximum(sps.factorial(l[:,None]-m[None,:]),1))
        if var.ndim == 4:
           return (var[None,None,:,:,:,:]*YdOmg[:,:,None,:,:,None]).sum(axis=(3,4))/intYdOmg[:,:,None,None]
        elif var.ndim == 3:
           return (var[None,None,:,:,:]*YdOmg[:,:,None,:,:]).sum(axis=(3,4))/intYdOmg[:,:,None]

    def spectral_decomposition_spin_weighted_2d(self,var):
        """
        spin weighted spectral decomposition in 2d for s=-2
        Args:
            @var (float): variable to decompose
        """
        self.touch()
        Y20=np.sqrt(15.0/(32.0*np.pi))*(np.sin(self.y))**3
        dOmg=np.sin(self.y[None,:])*self.dy[None,:]*self.dz[:,None]
        YdOmg=Y20[None,:]*dOmg
        intYdOmg=YdOmg.sum(axis=(0,1))
        return (var*YdOmg[None,:,:,None]).sum(axis=(1,2))/intYdOmg

    def check_bounce(self,ts,te):
        """
        check for bounce
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        rho = self.readvar('rho',tstart=ts,tend=te)
        s = self.readvar('s',tstart=ts,tend=te)
        rho_trap = 2e12 # 2e12 gm/cm3
        s_bounce = 3.0  # 3 kb/bary
        for t in range(ts,te+1):
            idx=np.where(rho[t-ts,:] > rho_trap)
            if np.any(s[t-ts][idx] > s_bounce):
                return t
        return None

#    def find_shock_position_index(self,ts,te,tb=None,idx=None):
#        """
#        find shock index
#        Args:
#            @te (int): ending time step
#            @ts (int): starting time step
#            @tb (Optional[int]): bounce time step
#        """
#        self.touch(time=te)
#        if tb == None:
#            tb == self.check_bounce(0,te)
#            if tb == None:
#                return None
#
#        if ts < tb:
#            ts = tb
#        Ma_shock = 1.0
#        if idx == None:
#            idx = 0
#        Ma = self.readvar('Ma',tstart=ts,tend=te)[:,:,:,idx:]
#        return np.apply_along_axis(lambda a: int(find_1d(a,Ma_shock,1,self.nx[0])), axis = Ma.ndim-1, arr = Ma)
#
    def find_shock_position_index(self,ts,te,tb=None,xn=None,vel=None,s=None):
        """
        find shock index
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @tb  (Optional[int]): bounce time step
            @xn  (Optional[int]): Non-None=based on neutron fraction
            @vel (Optional[int]): Non-None=based on radial velocity
            @s   (Optional[int]): Non-None=based on entropy
        """
        self.touch(time=te)
        if tb == None:
            tb == self.check_bounce(0,te)
            if tb == None:
                return None

        if ts < tb:
            ts = tb

        if xn != None:
            idx = self.x.searchsorted(5100e5)
            xn = self.readvar('Xn',tstart=ts,tend=te)
            #dxn = abs((xn[:,:,:,1:self.nx[0]]-xn[:,:,:,0:self.nx[0]-1])/xn[:,:,:,1:self.nx[0]])
            dxn = abs((xn[:,:,:,1:idx]-xn[:,:,:,0:idx-1])/xn[:,:,:,1:idx])
            return np.argmax(dxn,axis=(3))

        if vel != None:
            vel_sh = 1000 # 500 km/s
            vel = self.readvar('velocity/velx',tstart=ts,tend=te)*self.c_light
            return np.apply_along_axis(lambda a: int(find_1d(a,vel_sh,3,self.nx[0]-1)), axis = vel.ndim-1, arr = vel)
            #return np.apply_along_axis(lambda a: int(find_1d(a,vel_sh,3,476)), axis = vel.ndim-1, arr = vel)

        if s != None:
            s_sh = 10 # 10 kb/bary
            s = self.readvar('s',tstart=ts,tend=te)
            return np.apply_along_axis(lambda a: int(find_1d(a,s_sh,3,self.nx[0]-1)), axis = s.ndim-1, arr = s)
            #return np.apply_along_axis(lambda a: int(find_1d(a,s_sh,3,476)), axis = s.ndim-1, arr = s)

        rho = self.readvar('rho',tstart=ts,tend=te)
        drho=abs((rho[:,:,:,1:self.nx[0]]-rho[:,:,:,0:self.nx[0]-1])/rho[:,:,:,1:self.nx[0]])
        return np.argmax(drho,axis=(3))

#    def find_shock_position_index(self,ts,te,tb=None,xn=None):
#        """
#        find shock index
#        Args:
#            @te (int): ending time step
#            @ts (int): starting time step
#            @tb (Optional[int]): bounce time step
#        """
#        self.touch(time=te)
#        if tb == None:
#            tb == self.check_bounce(0,te)
#            if tb == None:
#                return None
#
#        if ts < tb:
#            ts = tb
#        vr = self.readvar('velocity/velx',tstart=ts,tend=te)
#        vra = self.calculate_mass_average_quantity('velocity/velx',ts,te)
#        return np.argmax(np.abs(vr-vra[:,None,None,:]),axis=(3))

    def find_shock_radius(self,ts,te,tb=None,xn=None,vel=None,s=None,areal=None):
        """
        find shock position
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @tb (Optional[int]): bounce time step
            @xn  (Optional[int]): Non-None=based on neutron fraction
            @vel (Optional[int]): Non-None=based on radial velocity
            @s   (Optional[int]): Non-None=based on entropy
        """
        idx = self.find_shock_position_index(ts,te,tb,xn,vel,s)
        if areal == None:
            phi = self.readvar('phi',ts,te)
            gphixx = self.readvar('gphixx',ts,te)
            sqrtgxx = np.sqrt(np.exp(4.0*phi)*gphixx)
            dx = sqrtgxx[:,:,:,:]*self.dx[None,None,None,:].cumsum(axis=(3))
        else:
            dx = self.calculate_cell_areal_radius(ts,te)
        dx1=dx.reshape(dx.shape[0]*dx.shape[1]*dx.shape[2],dx.shape[3])
        idx1=idx.reshape(dx.shape[0]*dx.shape[1]*dx.shape[2])
        return dx1[np.arange(len(idx1)),idx1].reshape(dx.shape[0],dx.shape[1],dx.shape[2])

    def find_neutron_star_radius_index(self,ts,te,tb=None,rho_ns=None):
        """
        find neutron star radius index 
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @tb (Optional[int]): bounce time step
            @rho_ns (Optional[int]): surface density of neutron star
        """
        self.touch(time=te)
        if tb == None:
            tb == self.check_bounce(0,te)
            if tb == None:
                return None

        if ts < tb:
            ts = tb
        if rho_ns == None:
          rho_ns = 1e11
        rho = self.readvar('rho',tstart=ts,tend=te)
        return np.apply_along_axis(lambda a: int(find_1d(a,rho_ns,0,self.nx[0])), axis = rho.ndim-1, arr = rho)

    def find_neutron_star_radius(self,ts,te,tb=None,rho_ns=None,areal=None):
        """
        find neutron star radius
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @tb (Optional[int]): bounce time step
            @rho_ns (Optional[int]): surface density of neutron star
        """
        idx = self.find_neutron_star_radius_index(ts,te,tb,rho_ns)
        if areal == None:
            phi = self.readvar('phi',ts,te)
            gphixx = self.readvar('gphixx',ts,te)
            sqrtgxx = np.sqrt(np.exp(4.0*phi)*gphixx)
            dx = sqrtgxx[:,:,:,:]*self.dx[None,None,None,:].cumsum(axis=(3))
        else:
            dx = self.calculate_cell_areal_radius(ts,te)
        dx1=dx.reshape(dx.shape[0]*dx.shape[1]*dx.shape[2],dx.shape[3])
        idx1=idx.reshape(dx.shape[0]*dx.shape[1]*dx.shape[2])
        return dx1[np.arange(len(idx1)),idx1].reshape(dx.shape[0],dx.shape[1],dx.shape[2])


    def calculate_neutron_star_mass(self,ts,te,tb=None,rho_ns=None):
        """
        calculate neutron star mass
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @tb (Optional[int]): bounce time step
            @rho_ns (Optional[int]): surface density of neutron star
        """
        self.touch(time=te)
        if tb == None:
            tb == self.check_bounce(0,te)
            if tb == None:
                return None

        if ts < tb:
            ts = tb

        mc=self.calculate_mass_coordinate(ts,te)
        rns=self.find_neutron_star_radius_index(ts,te,tb,rho_ns)
        if rns.ndim == 4: 
            arns=rns.mean(axis=(1,2,3),dtype=int)
        elif rns.ndim == 3: 
            arns=rns.mean(axis=(1,2),dtype=int)
        return mc[np.arange(len(arns)),arns]

    def calculate_neutron_star_lateral_kinetic_energy(self,ts,te,tb=None):
        """
        calculate neutron star lateral kinetic energy
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        if tb == None:
            tb == self.check_bounce(0,te)
            if tb == None:
                return None

        if ts < tb:
            ts = tb

        E=self.calculate_cell_lateral_kinetic_energy(ts,te).sum(axis=(1,2)).cumsum(axis=(1))
        rns=self.find_neutron_star_radius_index(ts,te,tb=tb)
        arns=rns.mean(axis=(1,2),dtype=int)
        return E[np.arange(len(arns)),arns]

    def calculate_neutron_star_rotational_period(self,ts,te,tb=None):
        """
        calculate neutron star rotational period
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        if tb == None:
            tb == self.check_bounce(0,te)
            if tb == None:
                return None

        if ts < tb:
            ts = tb

        arns=self.find_neutron_star_radius_index(ts,te,tb=tb)
        mass=self.calculate_cell_mass(ts,te)
        rad=self.calculate_radial_distance(ts,te)
        mr2=mass*rad**2
        jz=self.calculate_cell_angular_momentum(ts,te)
        Ins=self.calculate_neutron_star_sum(arns=arns,var=mr2).sum(axis=(1,2))
        jzns=self.calculate_neutron_star_sum(arns=arns,var=jz).sum(axis=(1,2))
        return 2.0*np.pi*Ins/jzns

    def calculate_neutron_star_sum(self,arns,var,time=None):
        """
        sum over neutron star
        Args:
            @arns (int): average neutron star radius index
            @var (float): variable to sum
        """
        self.touch(time=time)
        if var.ndim==2:
            varg=lambda a: var[a,0:arns[a]]
            return np.array([varg(i).sum(axis=(0)) for i in range(0,var.shape[0])],dtype='float64')
        elif var.ndim==3:
            varg=lambda a,b: var[a,b,0:arns[b]]
            return np.array([[varg(i,j).sum(axis=(0)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])],dtype='float64')
        elif var.ndim==4:
            varg=lambda a,b,c: var[a,b,c,0:arns[a,b,c]]
            return np.array([[[varg(i,j,k).sum(axis=(0)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])] for k in range(0,var.shape[2])],dtype='float64').T

    def calculate_neutron_star_mean(self,arns,var,time=None):
        """
        mean over neutron star
        Args:
            @arns (int): average neutron star radius index
            @var (float): variable to mean
        """
        self.touch(time=time)
        if var.ndim==2:
            varg=lambda a: var[a,0:arns[a]]
            return np.array([varg(i).mean(axis=(0)) for i in range(0,var.shape[0])],dtype='float64')
        elif var.ndim==3:
            varg=lambda a,b: var[a,b,0:arns[a]]
            return np.array([[varg(i,j).mean(axis=(0)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])],dtype='float64')
        elif var.ndim==4:
            varg=lambda a,b,c: var[a,b,c,0:arns[a,b,c]]
            return np.array([[[varg(i,j,k).mean(axis=(0)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])] for k in range(0,var.shape[2])],dtype='float64').T

    def find_quark_star_radius_index(self,ts,te,e):
        """
        find neutron star radius index 
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @e (class): eos class
        """
        self.touch(time=te)
        rho = self.readvar('rho',tstart=ts,tend=te)
        temp = self.readvar('temp',tstart=ts,tend=te)
        ye = self.readvar('ye',tstart=ts,tend=te)
        result = e.readeos(rho = rho.reshape((rho.shape[0] * rho.shape[1] * rho.shape[2] * rho.shape[3])),
            temp = temp.reshape((rho.shape[0] * rho.shape[1] * rho.shape[2] * rho.shape[3])),
            ye = ye.reshape((rho.shape[0] * rho.shape[1] * rho.shape[2] * rho.shape[3])))
        qvol = result[:, -1].reshape((rho.shape[0], rho.shape[1], rho.shape[2], rho.shape[3]))
        return qvol.shape[-1] - np.apply_along_axis(lambda a: a.searchsorted(0.99), axis = qvol.ndim-1, arr = qvol[:, :, :, ::-1])

    def find_quark_star_radius(self,ts,te,e):
        """
        find neutron star radius
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @e (class): eos class
        """
        idx = self.find_quark_star_radius_index(ts,te,e)
        phi = self.readvar('phi',ts,te)
        gphixx = self.readvar('gphixx',ts,te)
        sqrtgxx = np.sqrt(np.exp(4.0*phi)*gphixx)
        dx = sqrtgxx[:,:,:,:]*self.dx[None,None,None,:].cumsum(axis=(3))
        dx1 = dx.reshape(dx.shape[0]*dx.shape[1]*dx.shape[2],dx.shape[3])
        idx1 = idx.reshape(dx.shape[0]*dx.shape[1]*dx.shape[2])
        return dx1[np.arange(len(idx1)),idx1].reshape(dx.shape[0],dx.shape[1],dx.shape[2])

    def calculate_quark_star_mass(self,ts,te,e):
        """
        calculate neutron star mass
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @e (class): eos class
        """
        self.touch(time=te)
        rho = self.readvar('rho',tstart=ts,tend=te)
        temp = self.readvar('temp',tstart=ts,tend=te)
        ye = self.readvar('ye',tstart=ts,tend=te)
        result = e.readeos(rho = rho.reshape((rho.shape[0] * rho.shape[1] * rho.shape[2] * rho.shape[3])),
            temp=self.temp.reshape((rho.shape[0] * rho.shape[1] * rho.shape[2] * rho.shape[3])),
            ye=self.ye.reshape((rho.shape[0] * rho.shape[1] * rho.shape[2] * rho.shape[3])))
        qvol = result[:, -1].reshape((rho.shape[0] * rho.shape[1] * rho.shape[2] * rho.shape[3]))
        mass = self.calculate_cell_mass(ts,te)
        return (mass * qvol).sum(axis=(1, 2, 3))

    def calculate_luminosity(self,ts,te):
        """
        calculate total luminosity for each energy bin
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        L = self.readvar('L',tstart=ts,tend=te)
        Ltot = L.sum(axis=(1,2,3))
        return Ltot

    def calculate_total_luminosity(self,ts,te):
        """
        calculate total luminosity
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        Lmean = self.readvar('Lmean',tstart=ts,tend=te)
        Ltot = Lmean.sum(axis=(1,2,3))
        return Ltot

    def calculate_neutrino_heating(self,ts,te,sps=None):
        """
        calculate neutrino heating
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @sps (Optional[int]): None=return total heating
                                  Non-none=return heating every neutrino type
        """
        self.touch(time=te)
        J = self.readvar('J',tstart=ts,tend=te)
        Je = self.readvar('Je',tstart=ts,tend=te)
        ka = self.readvar('kappa_a',tstart=ts,tend=te)

        qe = -ka[:,:,:,:,:,:,:]*self.c_light*(Je[:,:,:,:,:,:,:]-J[:,:,:,:,:,:,:])*\
             self.de[None,None,None,None,None,:,None]*self.conv_MeVtoErg
        qsp = qe.sum(axis=(5)) 

        if sps==None:
            if self.numneu[0] == 3:
                return (qsp[:,:,:,:,:,0] + qsp[:,:,:,:,:,1] + 4.0*qsp[:,:,:,:,:,2])
            else:
                return (qsp[:,:,:,:,:,0] + qsp[:,:,:,:,:,1] + qsp[:,:,:,:,:,2] + qsp[:,:,:,:,:,3] + 2.0*qsp[:,:,:,:,:,4])
        else:
            return qsp

    def calculate_gain_radius(self,ts,te,arns,arsh,vol=None,idx=None):
        """
        calculate gain radius
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @arns (int): average neutron star radius index
            @arsh (int): average shock radius index
            @vol (float): volume of cell
            @idx (Optional[int]): None=return radius
                                  Non-none=return index
        """
        self.touch(time=te)
        qe=self.readvar('SE',ts,te).sum(axis=(1))

        if arns.ndim == 1:
            qe=(qe*vol).sum(axis=(1,2))
            q=lambda a: qe[a,arns[a]:arsh[a]]
            if idx == None:
                return np.array([self.x[int(find_1d(q(i),0.0,1,q(i).size)+arns[i])] for i in range(0,arns.shape[0])],dtype='float64').T
            else:
                return np.array([find_1d(q(i),0.0,1,q(i).size)+arns[i] for i in range(0,arns.shape[0])],dtype='int').T
        elif arns.ndim==3:
            q=lambda a,b,c: qe[a,b,c,arns[a,b,c]:arsh[a,b,c]]
            if idx == None:
                return np.array([[[self.x[int(find_1d(q(i,j,k),0.0,1,q(i,j,k).size)+arns[i,j,k])] for i in range(0,arns.shape[0])] for j in range(0,arns.shape[1])] for k in range(0,arns.shape[2])],dtype='float64').T
            else:
                return np.array([[[find_1d(q(i,j,k),0.0,1,q(i,j,k).size)+arns[i,j,k] for i in range(0,arns.shape[0])] for j in range(0,arns.shape[1])] for k in range(0,arns.shape[2])],dtype='int').T

    def calculate_gain_layer_heating(self,ts,te,arg,arsh,vol):
        """
        calculate gain layer heating
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @arg (int): average gain radius index
            @arsh (int): average shock radius index
            @vol (float): volume of cell
        """
        self.touch(time=te)
        qe=self.readvar('SE',ts,te).sum(axis=(1))
        #qe=(self.calculate_neutrino_heating(ts,te)).sum(axis=(1))
        qetot=(qe*vol).sum(axis=(1,2))
        q=lambda a: qetot[a,arg[a]:arsh[a]]
        return np.array([q(i).sum(axis=(0)) for i in range(0,arg.size)],dtype='float64')

    def calculate_cooling_layer_cooling(self,ts,te,arg,arns,vol):
        """
        calculate cooling layer cooling
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @arg (int): average gain radius index
            @arns (int): average NS radius index
            @vol (float): volume of cell
        """
        self.touch(time=te)
        qe=(self.calculate_neutrino_heating(ts,te)).sum(axis=(1))
        qetot=(qe*vol).sum(axis=(1,2))
        q=lambda a: qetot[a,arns[a]:arg[a]]
        return np.array([q(i).sum(axis=(0)) for i in range(0,arg.size)],dtype='float64')

    def calculate_gain_layer_heating_efficiency(self,ts,te,arg,arsh,q):
        """
        calculate gain layer heating
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @arg (int): average gain radius index
            @arsh (int): average shock radius index
            @q (float): gain layer heating
        """
        self.touch(time=te)
        Ltot=self.calculate_total_luminosity(ts,te) 
        L=lambda a: Ltot[a,arg[a]:arsh[a],0]+Ltot[a,arg[a]:arsh[a],1]
        Lg=np.array([L(i).mean(axis=(0)) for i in range(0,arg.size)],dtype='float64') 
        return q/Lg

    def calculate_cooling_layer_sum(self,arns,arg,var):
        """
        sum over gain layer
        Args:
            @arg (int): gain radius index
            @arns (int): neutron star radius index
            @var (float): variable to sum
        """
        self.touch()
        if var.ndim==2:
            varg=lambda a: var[a,arns[a]:arg[a]]
            return np.array([varg(i).sum(axis=(0)) for i in range(0,var.shape[0])],dtype='float64')
        elif var.ndim==3:
            varg=lambda a,b: var[a,b,arns[a]:arg[a]]
            return np.array([[varg(i,j).sum(axis=(0)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])],dtype='float64')
        elif var.ndim==4:
            varg=lambda a,b,c: var[a,b,c,arns[a,b,c]:arg[a,b,c]]
            return np.array([[[varg(i,j,k).sum(axis=(0)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])] for k in range(0,var.shape[2])],dtype='float64').T

    def calculate_gain_layer_sum(self,arg,arsh,var,ray=None):
        """
        sum over gain layer
        Args:
            @arg (int): average gain radius index
            @arsh (int): average shock radius index
            @var (float): variable to sum
        """
        self.touch()
        if var.ndim==2:
            varg=lambda a: var[a,arg[a]:arsh[a]]
            return np.array([varg(i).sum(axis=(0)) for i in range(0,var.shape[0])],dtype='float64')
        elif var.ndim==3:
            varg=lambda a,b: var[a,b,arg[b]:arsh[b]]
            return np.array([[varg(i,j).sum(axis=(0)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])],dtype='float64')
        elif var.ndim==4:
            varg=lambda a,b,c: var[a,b,c,arg[a,b,c]:arsh[a,b,c]]
            gain_sum = np.array([[[varg(i,j,k).sum(axis=(0)) for k in range(0,var.shape[2])] for j in range(0,var.shape[1])] for i in range(0,var.shape[0])],dtype='float64')
            if ray == None:
              gain_sum = gain_sum.sum(axis=(1,2))
            return gain_sum

    def calculate_gain_layer_mean(self,arg,arsh,var):
        """
        mean over gain layer
        Args:
            @arg (int): average gain radius index
            @arsh (int): average shock radius index
            @var (float): variable to mean
        """
        self.touch()
        if var.ndim==2:
            varg=lambda a: var[a,arg[a]:arsh[a]]
            return np.array([varg(i).mean(axis=(0)) for i in range(0,var.shape[0])],dtype='float64')
        elif var.ndim==3:
            varg=lambda a,b: var[a,b,arg[b]:arsh[b]]
            return np.array([[varg(i,j).mean(axis=(0)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])],dtype='float64')
        elif var.ndim==4:
            varg=lambda a,b,c: var[a,b,c,arg[a,b,c]:arsh[a,b,c]]
            return np.array([[[varg(i,j,k).mean(axis=(3)) for i in range(0,var.shape[0])] for j in range(0,var.shape[1])] for k in range(0,var.shape[2])],dtype='float64').T

    def calculate_optical_depth(self,ts,te,dr,p,mode=None,flux=None):
        """
        calculate optical depth
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @dr (float): cell radial length
            @p (float): power of energy
            @mode (int): None=absoption opacity
                         1=transport opacity kt=ka+ks
                         2=effective opacity keff=sqrt(ka*kt)
            @flux (Optional[int]): 0 = scale with total number flux H/e
                                   1 = scale with radial number flux H1/e
                                   2 = scale with polar number flux H2/e
                                   3 = scale with azimuthal number flux H3/e 
                                   None = scale with number density J/e
        """
        self.touch(time=te)
        if mode == None:
            k=self.energy_integrate('kappa_a',ts,te,p=p,flux=flux,scale=1)
        elif mode == 1:
            ka=self.energy_integrate('kappa_a',ts,te,p=p,flux=flux,scale=1)
            ks=self.energy_integrate('kappa_s',ts,te,p=p,flux=flux,scale=1)
            k = ka + ks
        elif mode == 2:
            ka=self.energy_integrate('kappa_a',ts,te,p=p,flux=flux,scale=1)
            ks=self.energy_integrate('kappa_s',ts,te,p=p,flux=flux,scale=1)
            kt = ka + ks
            k=np.sqrt(ka*kt)
        k=k*dr[:,None,:,:,:,None]
        return k[:,:,:,:,::-1,:].cumsum(axis=(4))[:,:,:,:,::-1,:]

    def find_neutrino_sphere_index(self,ts,te,dr,p,mode=None,flux=None):
        """
        find neutrino sphere radius
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @dr (float): cell radial length
            @p (float): power of energy
            @mode (int): None=absoption opacity
                         1=transport opacity kt=ka+ks
                         2=effective opacity keff=sqrt(ka*kt)
            @flux (Optional[int]): 0 = scale with total number flux H/e
                                   1 = scale with radial number flux H1/e
                                   2 = scale with polar number flux H2/e
                                   3 = scale with azimuthal number flux H3/e 
                                   None = scale with number density J/e
        """
        self.touch(time=te)
        tau=self.calculate_optical_depth(ts,te,dr,p=p,mode=mode,flux=flux)
        tau_nu_sph=2.0/3.0
        return np.apply_along_axis(lambda a: int(find_1d(a,tau_nu_sph,0,self.nx[0])), axis = 4, arr = tau)

    def find_neutrino_sphere_radius(self,ts,te,dr,p,mode=None,flux=None):
        """
        find neutrino sphere radius
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @dr (float): cell radial length
            @p (float): power of energy
            @mode (int): None=absoption opacity
                         1=transport opacity kt=ka+ks
                         2=effective opacity keff=sqrt(ka*kt)
            @flux (Optional[int]): 0 = scale with total number flux H/e
                                   1 = scale with radial number flux H1/e
                                   2 = scale with polar number flux H2/e
                                   3 = scale with azimuthal number flux H3/e 
                                   None = scale with number density J/e
        """
        self.touch(time=te)
        idx=self.find_neutrino_sphere_index(ts,te,dr,p=p,mode=mode,flux=flux).mean(axis=(1)).astype('int')
        r=dr.cumsum(axis=(3))
        idx1=idx.reshape(idx.shape[0]*idx.shape[1]*idx.shape[2],idx.shape[3])
        r1=r.reshape(r.shape[0]*r.shape[1]*r.shape[2],r.shape[3])
        rnu_e=r1[np.arange(len(idx1[:,0])),idx1[:,0]].reshape(r.shape[0],r.shape[1],r.shape[2])
        rnu_ae=r1[np.arange(len(idx1[:,1])),idx1[:,1]].reshape(r.shape[0],r.shape[1],r.shape[2])
        rnu_x=r1[np.arange(len(idx1[:,2])),idx1[:,2]].reshape(r.shape[0],r.shape[1],r.shape[2])
        return np.array([rnu_e.T,rnu_ae.T,rnu_x.T],dtype='float64').T

    def pairprocess_cooling(self,ts,te):
        """
        find neutrino sphere radius
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        e2de=self.e**2*self.de*self.conv_MeVtoErg**3 # erg**3
        e3de=self.e**3*self.de*self.conv_MeVtoErg**4 # erg**4
        e4de=self.e**4*self.de*self.conv_MeVtoErg**3 # erg**3*MeV**2
        temp=self.readvar('temp',ts,te)
        mu_e=self.readvar('mu_e',ts,te)
        expo_el=(self.e[:,None,None,None,None,None]-mu_e[None,:])/temp[None,:]
        expo_po=(self.e[:,None,None,None,None,None]+mu_e[None,:])/temp[None,:]
        FD_el=(8.0*np.pi/(self.h_planck*self.c_light)**3)/(1.0+np.exp(expo_el))
        FD_po=(8.0*np.pi/(self.h_planck*self.c_light)**3)/(1.0+np.exp(expo_po))
        n_el=(e2de[:,None,None,None,None,None]*FD_el).sum(axis=(0))
        n_po=(e2de[:,None,None,None,None,None]*FD_po).sum(axis=(0))
        e_el=(e3de[:,None,None,None,None,None]*FD_el).sum(axis=(0))/n_el
        e_po=(e3de[:,None,None,None,None,None]*FD_po).sum(axis=(0))/n_po
        e2_el=(e4de[:,None,None,None,None,None]*FD_el).sum(axis=(0))/n_el
        e2_po=(e4de[:,None,None,None,None,None]*FD_po).sum(axis=(0))/n_po
        const1 = (1.0/36.0)*(self.sigma0*self.c_light/self.mec2**2)*(self.CA**2+self.CV**2)
        const2 = (1.0/48.0)*(self.sigma0*self.c_light)*(2.0*self.CV**2-self.CA**2)
        return n_el*n_po*(const1*(e2_el*e_po+e2_po*e_el)+const2*(e_el+e_po))

    def pairprocess_heating(self,ts,te):
        """
        find neutrino sphere radius
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        e2de=self.e**2*self.de*self.conv_MeVtoErg**3 # erg**3
        e3de=self.e**3*self.de*self.conv_MeVtoErg**4 # erg**4
        e4de=self.e**4*self.de*self.conv_MeVtoErg**3 # erg**3*MeV**2
        J=self.readvar('J',ts,te)
        f=self.readvar('fH1',ts,te)
        chi=self.readvar('chi11',ts,te)
        f[f>1.0]=1.0
        phi=(3.0/4.0)*(1.0-2.0*f[:,:,:,:,:,:,None,0]*f[:,:,:,:,:,None,:,1]\
        +chi[:,:,:,:,:,:,None,0]*chi[:,:,:,:,:,None,:,1]+\
        0.5*(1.0-chi[:,:,:,:,:,:,None,0])*(1.0-chi[:,:,:,:,:,None,:,1]))
        const=self.conv_MeVtoErg*(2.0/9.0)*(self.CV**2+self.CA**2)*(self.sigma0*self.c_light/self.mec2**2)*(self.e[:,None]+self.e[None,:])
        return (const[None,None,None,None,None,:,:]*phi*J[:,:,:,:,:,:,None,0]*J[:,:,:,:,:,None,:,1]).sum(axis=(5,6))

    def calculate_cell_vortex(self,ts,te):
        """
        calculate cell vortex
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        vx=self.readvar('velocity/velx',te,te)
        vy=self.readvar('velocity/vely',te,te)*self.x[None,None,None,:]
        dxdvy=(vy[:,:,:,1:]-vy[:,:,:,:-1])/(self.x[None,None,None,1:]-self.x[None,None,None,:-1])
        dxdvy=np.concatenate((np.zeros((vy.shape[0], vy.shape[1], vy.shape[2], 1)),dxdvy,np.zeros((vy.shape[0], vy.shape[1], vy.shape[2], 1))),axis=(3))
        dxdvy=0.5*(dxdvy[:,:,:,1:]+dxdvy[:,:,:,:-1])
        dydvx=(vx[:,:,1:,:]-vx[:,:,:-1,:])/(self.y[None,None,1:,None]-self.y[None,None,:-1,None])
        dydvx=np.concatenate((np.zeros((vx.shape[0], vx.shape[1], 1, vx.shape[3])),dydvx,np.zeros((vx.shape[0], vx.shape[1], 1, vx.shape[3]))),axis=(2))
        dydvx=0.5*(dydvx[:,:,1:,:]+dydvx[:,:,:-1,:])
        return self.c_light*(dxdvy-dydvx)/self.x[None,None,None,:]

    def poison_solver(self,S0,n=None):
        if n == None:
          n = 20

        l = np.linspace(0, n, n + 1)

        Yl = np.sin(self.y[None, None, :]) * self.dy[None, None, :] * self.dz[None, :, None] * \
            np.real(sps.sph_harm(0, l[:, None, None], self.z[None, :, None], self.y[None, None, :]))
        Al = Yl[:, :, :, None] * self.dx[None, None, None, :] * self.x[None, None, None, :]**(l[:, None, None, None] + 2)
        Cl = (Al[None, :, :, :, :] * S0[:, None, :, :, :]).cumsum(axis=-1).sum(axis=(2, 3))
        Bl = Yl[:, :, :, None] * self.dx[None, None, None, :] * self.x[None, None, None, :]**(1 - l[:, None, None, None])
        Dl = (Bl[None, :, :, :, :] * S0[:, None, :, :, :])[:, :, :, :, ::-1].cumsum(axis=-1)[:, :, :, :, ::-1].sum(axis=(2, 3))

        Yl = np.real(sps.sph_harm(0, l[:, None, None], self.z[None, :, None], self.y[None, None, :]))
        CYl = Cl[:, :, None, None, :] * Yl[None, :, :, :, None]
        DYl = Dl[:, :, None, None, :] * Yl[None, :, :, :, None]
                                                                                
        phi = - 4.0 * np.pi * self.g_grav * ((1.0 / (2.0*l[None, :, None, None, None] + 1)) * \
            (CYl / self.x[None, None, None, None, :]**(l[None, :, None, None, None]+1) + \
            DYl * self.x[None, None, None, None, :]**l[None, :, None, None, None]))

        return phi.sum(axis=1), phi[:, 0, :]

    def calculate_NE20(self,ts,te):
        """
        calculate gravitational wave amplitude time derivative NE20
        Args:
            @te (int): ending time step
            @ts (int): starting time step
        """
        self.touch(time=te)
        factor=(self.g_grav*8.0*np.sqrt(np.pi/15.0)/self.c_light**4)
        rho=self.readvar('rho',ts,te)
        vr=self.readvar('velocity/velx',ts,te)*self.c_light
        vt=self.readvar('velocity/vely',ts,te)*self.x[None,None,None,:]*self.c_light
        mass=self.calculate_cell_mass(ts,te)

        z=np.cos(self.y)
        z2=z**2
        frz=3.0*z2-1.0
        ftz=-3.0*z*np.sqrt(1.0-z2)

        fr=vr*frz[None,None,:,None]
        ft=vt*ftz[None,None,:,None]

        NE20=factor*mass*(fr+ft)*self.x[None,None,None,:] 
        return NE20.sum(axis=(1,2))

    def calculate_AE20(self,ts,te,nsph=None,norot=None):
        """
        calculate gravitational wave amplitude AE20
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @nsph  (Optional[int]): Non-none=non-spherical potential version (max. l)
            @norot (Optional[int]): Non-none=rotation is not considered
        """
        self.touch(time=te)
        factor=(self.g_grav*8.0*np.sqrt(np.pi/15.0)/self.c_light**4)
        rho=self.readvar('rho',ts,te)
        vr=self.readvar('velocity/velx',ts,te)*self.c_light
        vt=self.readvar('velocity/vely',ts,te)*self.x[None,None,None,:]*self.c_light
        if norot == None:
          vz=self.readvar('velocity/velz',ts,te)*self.x[None,None,None,:]*np.sin(self.y[None,None,:,None])*self.c_light
        else:
          vz = 0

        mass=self.calculate_cell_mass(ts,te)
        z=np.cos(self.y);z2=z**2;frz=3.0*z2-1;ftz=2.0-3.0*z2;frtz=-6.0*z*np.sqrt(1.0-z2)
        fr=vr**2*frz[None,None,:,None];ft=vt**2*ftz[None,None,:,None];fz=-vz**2;frt=vr*vt*frtz[None,None,:,None]

        alp=self.readvar('alp',ts,te)
        dalp=np.concatenate((np.zeros(alp[:,:,:,:1].shape),dxdf1(alp,self.x),np.zeros(alp[:,:,:,:1].shape)),axis=3)
        falp=-dalp*self.x[None,None,None,:]*frz[None,None,:,None]*self.c_light**2

        if nsph == None:
            AE20=factor*mass*(fr+ft+frt+falp)
        else:
            w=self.readvar('lorentz',ts,te)
            h=self.readvar('h',ts,te)/self.c_light**2
            p=self.readvar('p',ts,te)/self.c_light**2
            S0 = rho*h*w**2-p
            n = nsph

            l=np.linspace(0,n,n+1)
            lm1=np.linspace(-1,n-1,n+1)
            lm1[0]=0
            Yl=np.sin(self.y[None,None,:])*self.dy[None,None,:]*self.dz[None,:,None]*np.real(sps.sph_harm(0,l[:,None,None],self.z[None,:,None],self.y[None,None,:]))
            Al=Yl[:,:,:,None]*self.dx[None,None,None,:]*self.x[None,None,None,:]**(l[:,None,None,None]+2)
            Cl=(Al[None,:,:,:,:]*S0[:,None,:,:,:]).sum(axis=(2,3)).cumsum(axis=(2))
            Bl=Yl[:,:,:,None]*(self.dx[None,None,None,:]*self.x[None,None,None,:]**(1-l[:,None,None,None]))
            Dl=(Bl[None,:,:,:,:]*S0[:,None,:,:,:]).sum(axis=(2,3))[:,:,::-1].cumsum(axis=2)[:,:,::-1]

            rdphi=-4.0*np.pi*self.g_grav*((-(l[None,:,None]+1)*Cl/self.x[None,None,:]**(l[None,:,None]+1)+l[None,:,None]*Dl*self.x[None,None,:]**l[None,:,None])/(2*l[None,:,None]+1))[:,:,None,None,:]*\
                np.real(sps.sph_harm(0,l[:,None,None],self.z[None,:,None],self.y[None,None,:]))[None,:,:,:,None]
            rdphi0=rdphi[:,0,:]
            rdphi=rdphi.sum(axis=(1))
            drphi=-rdphi*frz[None,None,:,None]
            drphi0=-rdphi0*frz[None,None,:,None]

            scdphi = -4.0*np.pi*self.g_grav*(Cl/self.x[None,None,:]**(l[None,:,None]+1)+Dl*self.x[None,None,:]**l[None,:,None])*l[None,:,None]/(2*l[None,:,None]+1)
#            scdphi=scdphi[:,:,None,None,:]*(np.cos(self.y[None,None,None,:,None])**2/np.sin(self.y[None,None,None,:,None]))*\
#                (-np.cos(self.y[None,None,None,:,None])*\
#                np.real(sps.sph_harm(0,l[None,:,None,None,None],self.z[None,None,:,None,None],self.y[None,None,None,:,None])) + \
#                np.sqrt((2*l[None,:,None,None,None]+1)/np.maximum(1,2*l[None,:,None,None,None]-1))*\
#                np.real(sps.sph_harm(0,lm1[None,:,None,None,None],self.z[None,None,:,None,None],self.y[None,None,None,:,None])))

            scdphi=scdphi[:,:,None,None,:] * (1.0 / np.sin(self.y[None,None,None,:,None])) * \
                (l[None,:,None,None,None] * np.cos(self.y[None,None,None,:,None]) * \
                np.real(sps.sph_harm(0, l[None,:,None,None,None], self.z[None,None,:,None,None], self.y[None,None,None,:,None])) \
                - np.sqrt((2 * l[None,:,None,None,None] + 1) / np.maximum(1, 2 * l[None,:,None,None,None] - 1)) * \
                np.real(sps.sph_harm(0, lm1[None,:,None,None,None], self.z[None,None,:,None,None], self.y[None,None,None,:,None])))

            scdphi[:,0,:] = 0
            scdphi=scdphi.sum(axis=(1))
            dtphi=-0.5*scdphi*frtz[None,None,:,None]
            AE20=factor*mass*(fr+ft+frt+fz+falp+(drphi-drphi0+dtphi))

#            pot, pot0 = self.poison_solver(S0, n)
#            pot = pot - pot0
#            drdpot = np.concatenate((np.zeros(pot[:,:,:,:1].shape),dxdf1(pot,self.x),np.zeros(pot[:,:,:,:1].shape)),axis=-1)
#            dtdpot = np.concatenate((np.zeros(pot[:,:,:1,:].shape),dydf1(pot,self.y),np.zeros(pot[:,:,:1,:].shape)),axis=-2)
#
#            fdotSr_corr = -drdpot*self.x[None,None,None,:]*frz[None,None,:,None]
#            fdotSt_corr = dtdpot*-0.5*frtz[None,None,:,None]
#            AE20 = factor*mass*(fr+ft+frt+fz+falp)
#            AE20 = AE20 + factor*mass*(fdotSr_corr + fdotSt_corr)

        return AE20.sum(axis=(1,2))

    def inverse_metric(self,gxx,gxy,gxz,gyy,gyz,gzz):
        detg = gxx*gyy*gzz - gxx*gyz**2 - gxy**2*gzz + 2*gxy*gxz*gyz -gxz**2*gyy
        uxx = (-gyz*gyz + gyy*gzz)/detg
        uxy = ( gxz*gyz - gxy*gzz)/detg
        uxz = (-gxz*gyy + gxy*gyz)/detg
        uyy = (-gxz*gxz + gxx*gzz)/detg
        uyz = ( gxy*gxz - gxx*gyz)/detg
        uzz = (-gxy*gxy + gxx*gyy)/detg
        return uxx, uxy, uxz, uyy, uyz, uzz

    def orthonormal_to_coordinate_contravariant_vector(self,vx,vy,vz):
        vcx = vx
        vcy = vy / self.x[None, None, None, :]
        vcz = vz / (self.x[None, None, None, :] * np.sin(self.y[None, None, :, None]))
        return vcx, vcy, vcz

    def orthonormal_to_coordinate_covariant_tensor(self,phi,gxx,gxy,gxz,gyy,gyz,gzz):
        conf = np.exp(4.0*phi)
        gcxx = conf*gxx
        gcxy = conf*gxy*self.x[None,None,None,:]
        gcxz = conf*gxz*(self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gcyy = conf*gyy*(self.x[None,None,None,:]*self.x[None,None,None,:])
        gcyz = conf*gyz* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gczz = conf*gzz* \
              (self.x[None,None,None,:]*self.x[None,None,None,:]* \
              np.sin(self.y[None,None,:,None])*np.sin(self.y[None,None,:,None]))
        return gcxx, gcxy, gcxz, gcyy, gcyz, gczz

    def orthonormal_to_coordinate_contravariant_tensor(self,phi,gxx,gxy,gxz,gyy,gyz,gzz):
        conf = np.exp(-4.0*phi)
        gcxx = conf*gxx
        gcxy = conf*gxy/self.x[None,None,None,:]
        gcxz = conf*gxz/(self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gcyy = conf*gyy/(self.x[None,None,None,:]*self.x[None,None,None,:])
        gcyz = conf*gyz/ \
              (self.x[None,None,None,:]*self.x[None,None,None,:]*np.sin(self.y[None,None,:,None]))
        gczz = conf*gzz/ \
              (self.x[None,None,None,:]*self.x[None,None,None,:]* \
              np.sin(self.y[None,None,:,None])*np.sin(self.y[None,None,:,None]))
        return gcxx, gcxy, gczz, gcyy, gcyz, gczz

    def calculate_AE20_GR(self,ts,te,nsph=None):
        """
        calculate gravitational wave amplitude AE20
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @nsph (Optional[int]): Non-none=non-spherical potential version
        """
        self.touch(time=te)
        factor=(self.g_grav*8.0*np.sqrt(np.pi/15.0)/self.c_light**4)

        rho  = self.readvar('rho',ts,te)
        w    = self.readvar('lorentz',ts,te)
        h    = self.readvar('h',ts,te)
        p    = self.readvar('p',ts,te)
        alp  = self.readvar('alp',ts,te)
        phi  = self.readvar('phi',ts,te)
        mass = self.calculate_cell_mass(ts,te)

        gxx = self.readvar('gphixx',ts,te)
        gxy = self.readvar('gphixy',ts,te)
        gxz = self.readvar('gphixz',ts,te)
        gyy = self.readvar('gphiyy',ts,te)
        gyz = self.readvar('gphiyz',ts,te)
        gzz = self.readvar('gphizz',ts,te)

        gcxx, gcxy, gcxz, gcyy, gcyz, gczz = self.orthonormal_to_coordinate_covariant_tensor(phi,gxx,gxy,gxz,gyy,gyz,gzz)
        ucxx, ucxy, ucxz, ucyy, ucyz, uczz = self.inverse_metric(gcxx,gcxy,gcxz,gcyy,gcyz,gczz)

        bur = self.readvar('betau/betaux',ts,te)
        but = self.readvar('betau/betauy',ts,te)
        buz = self.readvar('betau/betauz',ts,te)
        br = gxx*bur + gxy*but + gxz*buz
        bt = gxy*bur + gyy*but + gyz*buz
        bz = gxz*bur + gyz*but + gzz*buz
        b2 = bur*br + but*bt + buz*bz
        bcr, bct, bcz = self.orthonormal_to_coordinate_contravariant_vector(bur,but,buz)

        g00 = -alp**2 + b2

        velr = self.readvar('velocity/velx',ts,te)
        velt = self.readvar('velocity/vely',ts,te)
        velz = self.readvar('velocity/velz',ts,te)

        vr = gxx*velr + gxy*velt + gxz*velz
        vt = gxy*velr + gyy*velt + gyz*velz
        vz = gxz*velr + gyz*velt + gzz*velz

        velr = velr - bcr / alp
        velt = velt - bct / alp
        velz = velz - bcz / alp

        z=np.cos(self.y);z2=z**2;frz=3.0*z2-1;ftz=2.0-3.0*z2;frtz=-6.0*z*np.sqrt(1.0-z2)
        fr=vr*velr*frz[None,None,:,None];ft=vt*velt*ftz[None,None,:,None];fz=-vz*velz;frt=vr*velt*frtz[None,None,:,None]

        ptmp = p / (rho*h)
        t00 = (w**2 - ptmp) / alp**2
        t0r = w**2*velr / alp + ptmp*bcr / alp**2
        t0t = w**2*velt / alp + ptmp*bct / alp**2
        t0z = w**2*velz / alp + ptmp*bcz / alp**2

        trr = w**2*velr*velr + ptmp * (ucxx - bcr*bcr / alp**2)
        trt = w**2*velr*velt + ptmp * (ucxy - bcr*bct / alp**2)
        trz = w**2*velr*velz + ptmp * (ucxz - bcr*bcz / alp**2)
        ttt = w**2*velt*velt + ptmp * (ucyy - bct*bct / alp**2)
        ttz = w**2*velt*velz + ptmp * (ucyz - bct*bcz / alp**2)
        tzz = w**2*velz*velz + ptmp * (uczz - bcz*bcz / alp**2)

        dg00 = np.concatenate((np.zeros(g00[:,:,:,:1].shape),dxdf1(g00,self.x),np.zeros(g00[:,:,:,:1].shape)),axis=3)
        dgxx = np.concatenate((np.zeros(gcxx[:,:,:,:1].shape),dxdf1(gcxx,self.x),np.zeros(gcxx[:,:,:,:1].shape)),axis=3)
        dgxy = np.concatenate((np.zeros(gcxy[:,:,:,:1].shape),dxdf1(gcxy,self.x),np.zeros(gcxy[:,:,:,:1].shape)),axis=3)
        dgxz = np.concatenate((np.zeros(gcxz[:,:,:,:1].shape),dxdf1(gcxz,self.x),np.zeros(gcxz[:,:,:,:1].shape)),axis=3)
        dgyy = np.concatenate((np.zeros(gcyy[:,:,:,:1].shape),dxdf1(gcyy,self.x),np.zeros(gcyy[:,:,:,:1].shape)),axis=3)
        dgyz = np.concatenate((np.zeros(gcyz[:,:,:,:1].shape),dxdf1(gcyz,self.x),np.zeros(gcyz[:,:,:,:1].shape)),axis=3)
        dgzz = np.concatenate((np.zeros(gczz[:,:,:,:1].shape),dxdf1(gczz,self.x),np.zeros(gczz[:,:,:,:1].shape)),axis=3)

        dbr = np.concatenate((np.zeros(bur[:,:,:,:1].shape),dxdf1(bur,self.x),np.zeros(bur[:,:,:,:1].shape)),axis=3)
        dbt = np.concatenate((np.zeros(but[:,:,:,:1].shape),dxdf1(but,self.x),np.zeros(but[:,:,:,:1].shape)),axis=3)
        dbz = np.concatenate((np.zeros(buz[:,:,:,:1].shape),dxdf1(buz,self.x),np.zeros(buz[:,:,:,:1].shape)),axis=3)

        dotSr = 0.5*t00*dg00 + (t0r*dbr + t0t*dbt + t0z*dbz) + \
            0.5*(trr*dgxx + ttt*dgyy + tzz*dgzz) + \
            (trt*dgxy + trz*dgxz + ttz*dgyz)
        fdotSr = dotSr*self.x[None,None,None,:]*frz[None,None,:,None]

        if nsph == None:
            AE20=factor*mass*h*alp*(w**2*(fr+ft+frt+fz)+fdotSr)
        else:
            S0 = (rho*h*w**2-p)/alp**2/self.c_light**2
            n = nsph
            pot, pot0 = self.poison_solver(S0, n)
            pot = pot - pot0
            drdpot = np.concatenate((np.zeros(pot[:,:,:,:1].shape),dxdf1(pot,self.x),np.zeros(pot[:,:,:,:1].shape)),axis=-1)
            dtdpot = np.concatenate((np.zeros(pot[:,:,:1,:].shape),dydf1(pot,self.y),np.zeros(pot[:,:,:1,:].shape)),axis=-2)

            fdotSr_corr = -drdpot*self.x[None,None,None,:]*frz[None,None,:,None]
            fdotSt_corr = dtdpot*-0.5*frtz[None,None,:,None]
            AE20 = factor*mass*h*alp*(w**2*(fr+ft+frt+fz)+fdotSr)
            AE20 = AE20 + factor*mass*(fdotSr_corr + fdotSt_corr)

        return AE20.sum(axis=(1,2))

    def calculate_newtonian_potential(self,ts,te,phi_out=None):
        """
        calculate Newtonian gravitational potential
        Args:
            @te (int): ending time step
            @ts (int): starting time step
            @nsph (Optional[int]): Non-none=non-spherical potential version
        """
        rho=self.calculate_vol_average_quantity('rho', ts, te)
        m=rho.shape[0]
        n=rho.shape[1]
        phiin=np.zeros([m,n+1])
        phiout=np.zeros([m,n+1])
        phi=np.zeros([m,n+1])

        x_r=self.dx.cumsum()
        x_l=x_r-self.dx

        phiin[:, 0] = 0 
        phiout[:, -1] = 0

        phiin[:, 1:] = (4.0 * np.pi * self.g_grav / 3.0) * rho[:,:] * (x_r[None,:]**3-x_l[None,:]**3)
        phiin=phiin.cumsum(axis=1)

        phiout[:, :-1] = (4.0 * np.pi * self.g_grav / 2.0) * rho[:,:] * (x_r[None,:]**2-x_l[None,:]**2)
        phiout=phiout[:,::-1].cumsum(axis=1)[:,::-1]

        phi[:,0] = -phiout[:,0]

        phi[:,1:] = -(phiin[:,1:] / x_r[None,:] + phiout[:,1:])
        
        if phi_out == None:
          return phi
        else:
          return -phiout
