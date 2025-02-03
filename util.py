'''
Created on May 5, 2017

@author: ninoy
'''
import numpy as np

def find_1d(arr,val,mode,nx):
    """
    find value in 1d array
    Args:
        @arr (float): 1d array
        @val (float): value
        @mode (int): 0 < val
                     1 > val 
        @nx (int): number of points
    """
    if mode == 0:
        for i in range(0,nx):
            if arr[i] < val:
                return i

    if mode == 1:
        for i in range(0,nx):
            if arr[i] > val:
                return i

    if mode == 2:
        for i in range(nx-1,0,-1):
            if arr[i] < val:
                return i

    if mode == 3:
        for i in range(nx-1,0,-1):
            if arr[i] > val:
                return i

    return 0#nx-1

def dxdf1(f,x):
    """
    numerical first derivative 
    Args:
        @f (float): discrete value of function
        @x (float): discrete value of grid point
    """
    if f.ndim == 1:
        return (f[2:] - f[:-2])/(x[2:] - x[:-2])
    elif f.ndim == 4:
        return (f[:,:,:,2:] - f[:,:,:,:-2])/(x[2:] - x[:-2])
    elif f.ndim == 5:
        return (f[:,:,:,:,2:] - f[:,:,:,:,:-2])/(x[2:] - x[:-2])

def dydf1(f,y):
    """
    numerical first derivative 
    Args:
        @f (float): discrete value of function
        @y (float): discrete value of grid point
    """
    if f.ndim == 4:
        return (f[:,:,2:,:] - f[:,:,:-2,:])/(y[None,None,2:,None] - y[None,None,:-2,None])
    elif f.ndim == 5:
        return (f[:,:,:,2:,:] - f[:,:,:,:-2,:])/(y[None,None,2:,None] - y[None,None,:-2,None])

def dxdf2(f,x):
    """
    numerical second derivative 
    Args:
        @f (float): discrete value of function
        @x (float): discrete value of grid point
    """
    n=x.size-1
    if f.ndim == 4:
        return 2.0*(f[:,:,:,0:n-2]/((x[0:n-2]-x[1:n-1])*(x[0:n-2]-x[2:n])) + \
               f[:,:,:,1:n-1]/((x[1:n-1]-x[0:n-2])*(x[1:n-1]-x[2:n])) + \
               f[:,:,:,2:n]/((x[2:n]-x[0:n-2])*(x[2:n]-x[1:n-1])))
    elif f.ndim == 5:
        return 2.0*(f[:,:,:,:,0:n-2]/((x[0:n-2]-x[1:n-1])*(x[0:n-2]-x[2:n])) + \
               f[:,:,:,:,1:n-1]/((x[1:n-1]-x[0:n-2])*(x[1:n-1]-x[2:n])) + \
               f[:,:,:,:,2:n]/((x[2:n]-x[0:n-2])*(x[2:n]-x[1:n-1])))
