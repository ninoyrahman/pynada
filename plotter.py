'''
Created on Feb 16, 2017

@author: ninoy
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from pynada import pynada
import numpy as np
#plt.rc('text', usetex=True)

class plotter(pynada):
    """
    class for plotting NADA output 
    """

    def __init__(self):
        super().__init__()
        self.__line = None
        self.__cont = None

    def plotx(self,varname,time,yidx,zidx,yyidx=None,eidx=None,spidx=None,**kwargs):
        """
        1d plot along x axis
        Args:
            @varname (str): variable name
            @time (int): time step
            @yidx (int): y-coordinate index
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
        """
        super().__readh5__(varname,time)
        if yyidx == None and eidx == None and spidx == None:
            plt.plot(self.x,self.var[zidx,yidx,:],**kwargs)
        elif eidx == None and spidx == None:
            plt.plot(self.x,self.var[yyidx,zidx,yidx,:],**kwargs)
        elif eidx == None:
            plt.plot(self.x,self.var[yyidx,zidx,yidx,:,spidx],**kwargs)
        else:
            plt.plot(self.x,self.var[yyidx,zidx,yidx,:,eidx,spidx],**kwargs)

    def plote(self,varname,time,xidx,yidx,zidx,yyidx,spidx,**kwargs):
        """
        1d plot along e axis
        Args:
            @varname (str): variable name
            @time (int): time step
            @xidx (int): x-coordinate index
            @yidx (int): y-coordinate index
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @spidx (Optional[int]): neutrino species index 
        """
        super().__readh5__(varname,time)
        plt.plot(self.e,self.var[yyidx,zidx,yidx,xidx,:,spidx],**kwargs)

    def ploty(self,varname,time,xidx,zidx,yyidx=None,eidx=None,spidx=None,**kwargs):
        """
        1d plot along y axis
        Args:
            @varname (str): variable name
            @time (int): time step
            @xidx (int): x-coordinate index
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
        """
        super().__readh5__(varname,time)
        if yyidx == None and eidx == None and spidx == None:
            plt.plot(self.y,self.var[zidx,:,xidx],**kwargs)
        elif eidx == None and spidx == None:
            plt.plot(self.y,self.var[yyidx,zidx,:,xidx],**kwargs)
        elif eidx == None:
            plt.plot(self.y,self.var[yyidx,zidx,:,xidx,spidx],**kwargs)
        else:
            plt.plot(self.y,self.var[yyidx,zidx,:,xidx,eidx,spidx],**kwargs)

    def plotz(self,varname,time,xidx,yidx,yyidx=None,eidx=None,spidx=None,**kwargs):
        """
        1d plot along z axis
        Args:
            @varname (str): variable name
            @time (int): time step
            @xidx (int): x-coordinate index
            @yidx (int): y-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
        """
        super().__readh5__(varname,time)
        if yyidx == None and eidx == None and spidx == None:
            plt.plot(self.z,self.var[:,yidx,xidx],**kwargs)
        elif eidx == None and spidx == None:
            plt.plot(self.z,self.var[yyidx,:,yidx,xidx],**kwargs)
        elif eidx == None:
            plt.plot(self.z,self.var[yyidx,:,yidx,xidx,spidx],**kwargs)
        else:
            plt.plot(self.z,self.var[yyidx,:,yidx,xidx,eidx,spidx],**kwargs)

    def plott(self,varname,xidx,yidx,zidx,yyidx=None,eidx=None,spidx=None,tstart=None,tend=None,**kwargs):
        """
        plot time evolution of variable
        Args:
            @varname (str): variable name
            @xidx (int): x-coordinate index
            @yidx (int): y-coordinate index
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
            @tstart (Optional[int]): starting time
            @tend (Optional[int]): ending time
        """
        number_of_file = len(super().__getfile__())
        time=[]
        func=[]
        if tstart==None and tend==None:
            for t in range(super().__gett0__(), number_of_file+super().__gett0__()):
                super().__readh5__(varname,t)
                if yyidx == None and eidx == None and spidx == None:
                    func.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])
                time.append(self.time)
        elif tstart==None and tend!=None:
            for t in range(super().__gett0__(), tend+1):
                super().__readh5__(varname,t)
                if yyidx == None and eidx == None and spidx == None:
                    func.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])
                time.append(self.time)
        elif tstart!=None and tend==None:
            for t in range(tstart, number_of_file+super().__gett0__()-tstart):
                super().__readh5__(varname,t)
                if yyidx == None and eidx == None and spidx == None:
                    func.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])
                time.append(self.time)
        elif tstart!=None and tend!=None:
            for t in range(tstart, tend+1):
                super().__readh5__(varname,t)
                if yyidx == None and eidx == None and spidx == None:
                    func.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])
                time.append(self.time)

        plt.plot(time,func,**kwargs)

    def plotmax(self,varname):
        """
        plot time evolution of max value of variable
        Args:
            @varname (str): variable name
        """
        number_of_file = len(super().__getfile__())
        time=[]
        func=[]
        for t in range(super().__gett0__(), number_of_file+super().__gett0__()):
            super().__readh5__(varname,t)
            func.append(self.var.value.max())
            time.append(self.time)
        plt.plot(time,func)

    def plotmin(self,varname):
        """
        plot time evolution of max value of variable
        Args:
            @varname (str): variable name
        """
        number_of_file = len(super().__getfile__())
        time=[]
        func=[]
        for t in range(super().__gett0__(), number_of_file+super().__gett0__()):
            super().__readh5__(varname,t)
            func.append(self.var.value.min())
            time.append(self.time)
        plt.plot(time,func)

    def plotxy(self,varname,time,zidx,yyidx=None,eidx=None,spidx=None,logaxis=None,**kwargs):
        """
        2d plot along xy plane
        Args:
            @varname (str): variable name
            @time (int): time step
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
            @logaxis (Optional[int]): log axis if logaxis is Non-None
        """
        super().__readh5__(varname,time)
        Fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        if logaxis != None:
            r, theta = np.meshgrid(np.log10(self.x), self.y)
        else:
            r, theta = np.meshgrid(self.x, self.y)
        if yyidx == None and eidx == None and spidx == None:
            cont=ax.contourf(theta, r, self.var[zidx,:,:],**kwargs)
        elif eidx == None and spidx == None:
            cont=ax.contourf(theta, r, self.var[yyidx,zidx,:,:],**kwargs)
        elif eidx == None:
            cont=ax.contourf(theta, r, self.var[yyidx,zidx,:,:,spidx],**kwargs)
        else:
            cont=ax.contourf(theta, r, self.var[yyidx,zidx,:,:,eidx,spidx],**kwargs)
        plt.colorbar(cont)

    def plotxz(self,varname,time,yidx,yyidx=None,eidx=None,spidx=None,logaxis=None,**kwargs):
        """
        2d plot along xz plane
        Args:
            @varname (str): variable name
            @time (int): time step
            @yidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
            @logaxis (Optional[int]): log axis if logaxis is Non-None
        """
        super().__readh5__(varname,time)
        plt.subplot(111,projection='polar')
        if logaxis != None:
            r, theta = np.meshgrid(np.log10(self.x), self.z)
        else:
            r, theta = np.meshgrid(self.x, self.z)
        if yyidx == None and eidx == None and spidx == None:
            cont=plt.contourf(theta, r, self.var[:,yidx,:],**kwargs)
        elif eidx == None and spidx == None:
            cont=plt.contourf(theta, r, self.var[yyidx,:,yidx,:],**kwargs)
        elif eidx == None:
            cont=plt.contourf(theta, r, self.var[yyidx,:,yidx,:,spidx],**kwargs)
        else:
            cont=plt.contourf(theta, r, self.var[yyidx,:,yidx,:,eidx,spidx],**kwargs)
        plt.colorbar(cont)

    def maxfunc(self,varname):
        """
        max value of a variable over all time step
        """
        super().__readh5__(varname,super().__gett0__())
        max = self.var.value.max()
        number_of_file = len(super().__getfile__())
        for t in range(super().__gett0__()+1, super().__gett0__()+number_of_file):
            super().__readh5__(varname,t)
            if max < self.var.value.max():
                max = self.var.value.max()
        return max
 
    def minfunc(self,varname):
        """
        min value of a variable over all time step
        """
        super().__readh5__(varname,super().__gett0__())
        min = self.var.value.min()
        number_of_file = len(super().__getfile__())
        for t in range(super().__gett0__()+1, super().__gett0__()+number_of_file):
            super().__readh5__(varname,t)
            if min > self.var.value.min():
                min = self.var.value.min()
        return min

    def __init(self):
        self.__line.set_data([], [])
        return self.__line,

    def __animate(self,i,varname,axis,iidx,jidx,yyidx=None,spidx=None,eidx=None):
        super().__readh5__(varname,i+super().__gett0__())
        if axis == 1:
            xc=self.x
            if yyidx == None and eidx == None and spidx == None:
                yc=self.var[jidx,iidx,:]
            elif eidx == None and spidx == None:
                yc=self.var[yyidx,jidx,iidx,:]
            elif eidx == None:
                yc=self.var[yyidx,jidx,iidx,:,spidx]
            else:
                yc=self.var[yyidx,jidx,iidx,:,eidx,spidx]
        if axis == 2:
            xc=self.x[iidx]*self.y
            if yyidx == None and eidx == None and spidx == None:
                yc=self.var[jidx,:,iidx]
            elif eidx == None and spidx == None:
                yc=self.var[yyidx,jidx,:,iidx]
            elif eidx == None:
                yc=self.var[yyidx,jidx,:,iidx,spidx]
            else:
                yc=self.var[yyidx,jidx,:,iidx,eidx,spidx]
        if axis == 3:
            xc=self.z
            if yyidx == None and eidx == None and spidx == None:
                yc=self.var[:,jidx,iidx]
            elif eidx == None and spidx == None:
                yc=self.var[yyidx,:,jidx,iidx]
            elif eidx == None:
                yc=self.var[yyidx,:,jidx,iidx,spidx]
            else:
                yc=self.var[yyidx,:,jidx,iidx,eidx,spidx]
        if axis == 4:
            xc=self.e
            yc=self.var[yyidx,eidx,jidx,iidx,:,spidx]
        self.__line.set_data(xc, yc)
        return self.__line,
    
    def animex(self,varname,yidx,zidx,yyidx=None,eidx=None,spidx=None):
        """
        1d anime along x axis
        Args:
            @varname (str): variable name
            @yidx (int): y-coordinate index
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
        """
        super().__readh5__(varname,super().__gett0__())
        fig = plt.figure()
        ax = plt.axes(xlim=(self.x.min(), self.x.max()), ylim=(self.minfunc(varname), self.maxfunc(varname)))
        self.__line, = ax.plot([], [], lw=2)
        number_of_file = len(super().__getfile__())
        if yyidx == None and eidx == None and spidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,1,yidx,zidx),\
                frames=number_of_file, interval=500, blit=True)
        elif eidx == None and spidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,1,yidx,zidx,yyidx),\
                frames=number_of_file, interval=500, blit=True)
        elif eidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,1,yidx,zidx,yyidx,spidx),\
                frames=number_of_file, interval=500, blit=True)
        else:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,1,yidx,zidx,yyidx,spidx,eidx),\
                frames=number_of_file, interval=500, blit=True)
        plt.grid()
        plt.show()

    def animee(self,varname,xidx,yidx,zidx,yyidx,spidx):
        """
        1d anime along e axis
        Args:
            @varname (str): variable name
            @xidx (int): x-coordinate index
            @yidx (int): y-coordinate index
            @zidx (int): z-coordinate index
            @yyidx (int): yinyang patch index
            @spidx (int): neutrino species index 
        """
        super().__readh5__(varname,super().__gett0__())
        fig = plt.figure()
        ax = plt.axes(xlim=(self.e.min(), self.e.max()), ylim=(self.minfunc(varname), self.maxfunc(varname)))
        self.__line, = ax.plot([], [], lw=2)
        number_of_file = len(super().__getfile__())
        anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,4,xidx,yidx,yyidx,spidx,zidx),\
            frames=number_of_file, interval=500, blit=True)
        plt.grid()
        plt.show()

    def animey(self,varname,xidx,zidx,yyidx=None,eidx=None,spidx=None):
        """
        1d anime along y axis
        Args:
            @varname (str): variable name
            @xidx (int): x-coordinate index
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
        """
        super().__readh5__(varname,super().__gett0__())
        fig = plt.figure()
        ax = plt.axes(xlim=(self.x[xidx]*self.y.min(), self.x[xidx]*self.y.max()), ylim=(self.minfunc(varname), self.maxfunc(varname)))
        self.__line, = ax.plot([], [], lw=2)
        number_of_file = len(super().__getfile__())
        if yyidx == None and eidx == None and spidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,2,xidx,zidx),\
                frames=number_of_file, interval=500, blit=True)
        elif eidx == None and spidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,2,xidx,zidx,yyidx),\
                frames=number_of_file, interval=500, blit=True)
        elif eidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,2,xidx,zidx,yyidx,spidx),\
                frames=number_of_file, interval=500, blit=True)
        else:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,2,xidx,zidx,yyidx,spidx,eidx),\
                frames=number_of_file, interval=500, blit=True)
        plt.grid()
        plt.show()

    def animez(self,varname,xidx,yidx,yyidx=None,eidx=None,spidx=None):
        """
        1d anime along z axis
        Args:
            @varname (str): variable name
            @xidx (int): x-coordinate index
            @yidx (int): y-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
        """
        super().__readh5__(varname,super().__gett0__())
        fig = plt.figure()
        ax = plt.axes(xlim=(self.z.min(), self.z.max()), ylim=(self.minfunc(varname), self.maxfunc(varname)))
        self.__line, = ax.plot([], [], lw=2)
        number_of_file = len(super().__getfile__())
        if yyidx == None and eidx == None and spidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,3,xidx,yidx),\
                frames=number_of_file, interval=500, blit=True)
        elif eidx == None and spidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,3,xidx,yidx,yyidx),\
                frames=number_of_file, interval=500, blit=True)
        elif eidx == None:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,3,xidx,yidx,yyidx,spidx),\
                frames=number_of_file, interval=500, blit=True)
        else:
            anim = animation.FuncAnimation(fig, self.__animate, init_func=self.__init,fargs=(varname,3,xidx,yidx,yyidx,spidx,eidx),\
                frames=number_of_file, interval=500, blit=True)
        plt.grid()
        plt.show()


    def __animate2d(self,i,varname,axis,iidx,yyidx=None,spidx=None,eidx=None):   
        super().__readh5__(varname,i+super().__gett0__())
        #plt.subplot(111)
        plt.subplot(111,projection='polar')
        if axis == 1:
            r, theta = np.meshgrid(self.x, self.y)
            if yyidx == None and eidx == None and spidx == None:
                self.__line = plt.contourf(theta, r, self.var[iidx,:,:],norm = LogNorm())
            elif eidx == None and spidx == None:
                self.__line = plt.contourf(theta, r, self.var[yyidx,iidx,:,:])
            elif eidx == None:
                self.__line = plt.contourf(theta, r, self.var[yyidx,iidx,:,:,spidx])
            else:
                self.__line = plt.contourf(theta, r, self.var[yyidx,iidx,:,:,eidx,spidx])
                #self.__line = plt.contourf(theta, r, self.var[yyidx,iidx,:,:,eidx,spidx],norm = LogNorm())
        if axis == 2:
            r, theta = np.meshgrid(self.x, self.z)
            if yyidx == None and eidx == None and spidx == None:
                self.__line = plt.contourf(theta, r, self.var[:,iidx,:],norm = LogNorm())
            elif eidx == None and spidx == None:
                self.__line = plt.contourf(theta, r, self.var[yyidx,:,iidx,:],norm = LogNorm())
            elif eidx == None:
                self.__line = plt.contourf(theta, r, self.var[yyidx,:,iidx,:,spidx],norm = LogNorm())
            else:
                self.__line = plt.contourf(theta, r, self.var[yyidx,:,iidx,:,eidx,spidx],norm = LogNorm())
        #plt.title(str(self.time.value))
        return self.__line,

    def animexy(self,varname,zidx,yyidx=None,eidx=None,spidx=None):
        """
        2d anime along xy plane
        Args:
            @varname (str): variable name
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
        """
        super().__readh5__(varname,super().__gett0__())
        fig = plt.figure()
        plt.subplot(111)
        plt.subplot(111,projection='polar')
        r, theta = np.meshgrid(self.x, self.y)
        number_of_file = len(super().__getfile__())
        if yyidx == None and eidx == None and spidx == None:
            self.__line = plt.contourf(theta, r, self.var[zidx,:,:])
            #self.__line = plt.contourf(theta, r, self.var[zidx,:,:],norm = LogNorm())
            anim = animation.FuncAnimation(fig, self.__animate2d, fargs=(varname,1,zidx),\
                frames=number_of_file, interval=1, blit=False)
        elif eidx == None and spidx == None:
            self.__line = plt.contourf(theta, r, self.var[yyidx,zidx,:,:])
            anim = animation.FuncAnimation(fig, self.__animate2d, fargs=(varname,1,zidx,yyidx),\
                frames=number_of_file, interval=1, blit=False)
        elif eidx == None:
            self.__line = plt.contourf(theta, r, self.var[yyidx,zidx,:,:,spidx])
            anim = animation.FuncAnimation(fig, self.__animate2d, fargs=(varname,1,zidx,yyidx,spidx),\
                frames=number_of_file, interval=1, blit=False)
        else:
            self.__line = plt.contourf(theta, r, self.var[yyidx,zidx,:,:,eidx,spidx])
            #self.__line = plt.contourf(theta, r, self.var[yyidx,zidx,:,:,eidx,spidx],norm = LogNorm())
            anim = animation.FuncAnimation(fig, self.__animate2d, fargs=(varname,1,zidx,yyidx,spidx,eidx),\
                frames=number_of_file, interval=1, blit=False)
        plt.colorbar()
        plt.show()

    def animexz(self,varname,yidx,yyidx=None,eidx=None,spidx=None):
        """
        2d anime along xz plane
        Args:
            @varname (str): variable name
            @yidx (int): y-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index
        """
        super().__readh5__(varname,super().__gett0__())
        fig = plt.figure()
        plt.subplot(111,projection='polar')
        r, theta = np.meshgrid(self.x, self.z)
        number_of_file = len(super().__getfile__())
        if yyidx == None and eidx == None and spidx == None:
            self.__line = plt.contourf(theta, r, self.var[:,yidx,:],norm = LogNorm())
            anim = animation.FuncAnimation(fig, self.__animate2d, fargs=(varname,2,yidx),\
                frames=number_of_file, interval=1, blit=False)
        elif eidx == None and spidx == None:
            self.__line = plt.contourf(theta, r, self.var[yyidx,:,yidx,:],norm = LogNorm())
            anim = animation.FuncAnimation(fig, self.__animate2d, fargs=(varname,2,yidx,yyidx),\
                frames=number_of_file, interval=1, blit=False)
        elif eidx == None:
            self.__line = plt.contourf(theta, r, self.var[yyidx,:,yidx,:,spidx],norm = LogNorm())
            anim = animation.FuncAnimation(fig, self.__animate2d, fargs=(varname,2,yidx,yyidx,spidx),\
                frames=number_of_file, interval=1, blit=False)
        else:
            self.__line = plt.contourf(theta, r, self.var[yyidx,:,yidx,:,eidx,spidx],norm = LogNorm())
            anim = animation.FuncAnimation(fig, self.__animate2d, fargs=(varname,2,yidx,yyidx,spidx,eidx),\
                frames=number_of_file, interval=1, blit=False)
        plt.colorbar()
        plt.show()

    def plotdatafile(self,filename,c1,c2,**kwargs):
        """
        2d anime along xz plane
        Args:
            @filename (str): file name
            @c1 (int): column in x axis
            @c2 (int): colume in y axis
        """
        f = open(filename, 'r')
        x = []
        y = []
        for line in f:
            line = line.strip()
            columns = line.split()
            x.append(float(columns[c1]))
            y.append(float(columns[c2]))
        plt.plot(x,y,**kwargs)
        f.close()

    def plotvv(self,varname1,varname2,xidx,yidx,zidx,yyidx=None,eidx=None,spidx=None,tstart=None,tend=None,**kwargs):
        """
        plot time evolution of variable
        Args:
            @varname1 (str): variable name along x axis
            @varname2 (str): variable name along y axis
            @xidx (int): x-coordinate index
            @yidx (int): y-coordinate index
            @zidx (int): z-coordinate index
            @yyidx (Optional[int]): yinyang patch index
            @eidx (Optional[int]): neutrino energy index
            @spidx (Optional[int]): neutrino species index 
            @tstart (Optional[int]): starting time
            @tend (Optional[int]): ending time
        """
        number_of_file = len(super().__getfile__())
        func1=[]
        func2=[]
 
        if tstart==None and tend==None:
            for t in range(super().__gett0__(), number_of_file+super().__gett0__()):
                super().__readh5__(varname1,t)
                if yyidx == None and eidx == None and spidx == None:
                    func1.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func1.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func1.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func1.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])

                super().__readh5__(varname2,t)
                if yyidx == None and eidx == None and spidx == None:
                    func2.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func2.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func2.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func2.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])
        elif tstart==None and tend!=None:
            for t in range(super().__gett0__(), tend+1):
                super().__readh5__(varname1,t)
                if yyidx == None and eidx == None and spidx == None:
                    func1.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func1.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func1.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func1.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])

                super().__readh5__(varname2,t)
                if yyidx == None and eidx == None and spidx == None:
                    func2.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func2.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func2.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func2.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])
        elif tstart!=None and tend==None:
            for t in range(tstart, number_of_file+super().__gett0__()-tstart):
                super().__readh5__(varname1,t)
                if yyidx == None and eidx == None and spidx == None:
                    func1.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func1.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func1.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func1.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])

                super().__readh5__(varname2,t)
                if yyidx == None and eidx == None and spidx == None:
                    func2.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func2.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func2.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func2.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])
        elif tstart!=None and tend!=None:
            for t in range(tstart, tend+1):
                super().__readh5__(varname1,t)
                if yyidx == None and eidx == None and spidx == None:
                    func1.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func1.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func1.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func1.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])

                super().__readh5__(varname2,t)
                if yyidx == None and eidx == None and spidx == None:
                    func2.append(self.var[zidx,yidx,xidx])
                elif eidx == None and spidx == None:
                    func2.append(self.var[yyidx,zidx,yidx,xidx])
                elif eidx == None:
                    func2.append(self.var[yyidx,zidx,yidx,xidx,spidx])
                else:
                    func2.append(self.var[yyidx,zidx,yidx,xidx,eidx,spidx])

        plt.plot(func1,func2,**kwargs)

