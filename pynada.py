'''
Created on Jun 8, 2016

@author: ninoy
'''
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np

class pynada:
    """
    class for data analysis of NADA output 
    Attributes:
        filepath (str) : path to NADA output files
        timesteps (int): total number of time step = number of files in  filepath.
        nx (int): number of cells in radial grid
        ny (int): number of cells in polar grid
        nz (int): number of cells in azimuthal grid
        ne (int): number of cells in neutrino energy grid
        numneu (int): number of neutrino species
        numpatch (int): number of yinyang patch
        x (double): radial grid
        y (double): polar grid
        z (double): azimuthal grid
        e (double): neutrino energy grid
        var (double) :: variable to analysis or visualize
    """
   
    def __init__(self):
        self = self
        self.filepath = None
        self.timesteps = None
        self.time = None
        self.__file = None
        self.__h5file = None
        self.__t0 = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.ne = None
        self.numneu = None
        self.numpatch = None
        self.x = None
        self.y = None
        self.z = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.e = None
        self.de = None
        self.yyw = None
        self.yym22 = None
        self.yym23 = None
        self.var = None

    def __getfile__(self):
        return self.__file

    def __geth5file__(self):
        return self.__h5file

    def __gett0__(self):
        return self.__t0


    def read(self,filepath):
        """
        read files from folder
        Args:
            @filepath (str): path to files
        """
        self.filepath = filepath
        self.__file = [f for f in listdir(filepath) if f.endswith('.h5')]     
        self.timesteps = len(self.__file)-1
        self.__t0 = int((min(self.__file).replace('fields_','')).replace('.h5',''))

    def update(self):
        """
        update the files
        """
        self.__file = None
        self.__file = [f for f in listdir(self.filepath) if f.endswith('.h5')]     
        self.timesteps = len(self.__file)-1
        self.__t0 = int((min(self.__file).replace('fields_','')).replace('.h5',''))

    def __readh5__(self,varname=None,time=None):
        self.update()
        if time==None:
            time = self.timesteps
        timestr = str(time).zfill(10)
        if varname!=None:
            varname = '/fields/'+varname
        for s in self.__file:
            if timestr in s:
                self.__h5file=h5py.File(self.filepath+'/'+s,'r')
                if '/grid/time' in self.__h5file:
                    self.time = np.array(self.__h5file['/grid/time'],dtype='float64')
                if '/setup/nx' in self.__h5file:
                    self.nx = np.array(self.__h5file['/setup/nx'])
                if '/setup/ny' in self.__h5file:
                    self.ny = np.array(self.__h5file['/setup/ny'])
                if '/setup/nz' in self.__h5file:
                    self.nz = np.array(self.__h5file['/setup/nz'])
                if '/setup/ne' in self.__h5file:
                    self.ne = np.array(self.__h5file['/setup/ne'])
                if '/setup/numneu' in self.__h5file:
                    self.numneu = np.array(self.__h5file['/setup/numneu'])
                if '/setup/numpatch' in self.__h5file:
                    self.numpatch = np.array(self.__h5file['/setup/numpatch'])
                if '/grid/x' in self.__h5file:
                    self.x = np.array(self.__h5file['/grid/x'],dtype='float64')
                if '/grid/y' in self.__h5file:
                    self.y = np.array(self.__h5file['/grid/y'],dtype='float64')
                if '/grid/z' in self.__h5file:
                    self.z = np.array(self.__h5file['/grid/z'],dtype='float64')
                if '/grid/dx' in self.__h5file:
                    self.dx = np.array(self.__h5file['/grid/dx'],dtype='float64')
                if '/grid/dy' in self.__h5file:
                    self.dy = np.array(self.__h5file['/grid/dy'],dtype='float64')
                if '/grid/dz' in self.__h5file:
                    self.dz = np.array(self.__h5file['/grid/dz'],dtype='float64')
                if '/grid/e' in self.__h5file:
                    self.e = np.array(self.__h5file['/grid/e'],dtype='float64')
                if '/grid/de' in self.__h5file:
                    self.de = np.array(self.__h5file['/grid/de'],dtype='float64')
                if '/grid/yyweight' in self.__h5file:
                    self.yyw = np.array(self.__h5file['/grid/yyweight'],dtype='float64')
                if '/grid/yym22' in self.__h5file:
                    self.yym22 = np.array(self.__h5file['/grid/yym22'],dtype='float64')
                if '/grid/yym23' in self.__h5file:
                    self.yym23 = np.array(self.__h5file['/grid/yym23'],dtype='float64')
                if varname!=None:
                    if varname in self.__h5file:
                        #self.var = self.__h5file[varname]
                        self.var = np.array(self.__h5file[varname],dtype='float64')
                    else:
                        print (varname+' not found')
                self.__h5file.close()

    def __printname(self,name,obj):
        if isinstance(obj,h5py.Dataset):
            print(str(obj.parent.name),str(obj.name).replace(str(obj.parent.name),'').replace('/',''),str(obj.shape))

    def printattr(self,time):
        print('Group Dataset Shape')
        self.__readh5__(varname=None,time=time)
        self.__h5file.visititems(self.__printname)


    def touch(self,time=None):
        """
        read coordinate data points and number of data points
        Args:
            @time (Optional[int]): time
        """
        self.update()
        self.__readh5__(varname=None,time=time)

    def readvar(self,varname,tstart=None,tend=None,h5=None):
        """
        read variable
        Args:
            @varname (str): variable name
            @tstart (Optional[int]): starting time
            @tend (Optional[int]): ending time
            @h5 (Optional): None = output in numpy array
                            Non-None = output in h5       
        """
        self.update()
        number_of_file = len(self.__file)
        func=[]
        if tstart==None and tend==None:
            for t in range(self.__t0, number_of_file+self.__t0):
                self.__readh5__(varname,t)
                func.append(self.var)
        elif tstart==None and tend!=None:
            for t in range(self.__t0, tend+1):
                self.__readh5__(varname,t)
                func.append(self.var)
        elif tstart!=None and tend==None:
            for t in range(tstart, number_of_file+self.__t0-tstart):
                self.__readh5__(varname,t)
                func.append(self.var)
        elif tstart!=None and tend!=None:
            for t in range(tstart, tend+1):
                self.__readh5__(varname,t)
                func.append(self.var)
        if h5==None:
            return np.array(func,dtype='float64')
        else:
            return func

