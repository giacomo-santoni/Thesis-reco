#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:45:19 2021

@author: alessandro
"""

#%%Importing the necessary libraries
import numpy as np
#%%Utilities for generating masks

#function for returning the necessary output in the sequence algorithm
#+1 for quadratic residues, -1 for the others
def quad_res_new(x,p):
     x=x%p
     for y in range(2,p, 1):
        if ((y*y)%p==x):
            return +1
     return -1
 
#%% Generation of MURA masks
   
#generation the MURA mask according to the algorithm  
def gen_mura(p):
    #startig with a (p,p) square matrix initialized to 0
    aij=np.zeros((p,p))
    #with n.ndindex(*shape) we iterate over the indices of the array
    for i,j in np.ndindex(aij.shape):
        if i==0:
            aij[i][j]=0
        elif quad_res_new(i,p)*quad_res_new(j,p)==+1:
            aij[i][j]=+1
        if j==0 and i!=0:
            aij[i][j]=+1
    #print(aij)
    #draw_mura(aij)
    return aij

#%%

#a function to generate a random mask of rank p
def gen_random(p, balance=0.5):
    size=np.array((p,p))
    arr = np.random.uniform(size=size)
    nwanted = int( np.floor( size[0] * size[1] * balance ) )
    s = np.sort(arr.flatten())
    cut = np.mean(s[nwanted-1:nwanted+1])
    return (arr > cut).astype(int)
#%%Generation of odd rank mosaics (Bologna-type)

#generation of the Bologna mosaic: placing 4 next to each other and cutting 
#the first row and column
def gen_mosaic1(p):
    mura=gen_mura(p)
    mosaic=np.tile(mura, (2,2))
    mosaic=mosaic[1:,1:]
    return mosaic

#mosaic obtained by flipping the MURAs on both axes and tiling with the new matrix
#keeping the central cross fixed (accomplished by performing a regular tiling
# and cutting the last row and column)
#This gives the mosaic from gen_mosaic1 in an alternative way
def gen_mosaic2(p):
    mura=gen_mura(p)
    mura=np.flip(mura)
    mosaic=np.tile(mura, (2,2))
    mosaic=mosaic[:(p*2-1), :(p*2-1)]
    return mosaic

#%%Generation of eve rank mosaics (Lecce-type)

#generation of the basic pattern with the Lecce method
def gen_lecce(p):
    aij=gen_mura(p)
    #holes in the Bologna notation in this way
    aij[:,0]=0
    aij[0,:]=+1
    #inversion of the pattern in the Lecce notation
    # aij=np.where(aij==+1,2,aij)
    # aij=np.where(aij==0,+1,aij)
    # aij=np.where(aij==2,0,aij)
    # aij[:,0]=+1
    # aij[0,:]=0
    #print(aij)
    #draw_mura(aij)
    return aij

#generation of the even rank mosaic
def gen_lecce_mosaic(p):
    mura=gen_lecce(p)
    mosaic=np.tile(mura, (2,2))
    roll_num=(p-1) // 2
    mosaic= np.roll(mosaic,[roll_num,roll_num],[1,0])
    mosaic = abs(mosaic - 1)
    return mosaic

#%%Generation of the decoding matrix
def gen_decode(aij):
    g=np.zeros_like(aij)
    for i,j in np.ndindex(g.shape):
        if aij[i][j]==1 and i+j!=0:
            g[i][j]=+1
        elif aij[i][j]==0 and i+j!=0:
            g[i][j]=-1
        if i+j==0:
            g[i][j]=+1
    #print(g)
    return g
#convoluzioni
def convolve1(a):
    g=gen_decode(a)
    phi=np.zeros_like(a)
    p=np.size(a,1)
    b=(p-1)//2
    for l,k in np.ndindex(phi.shape):
        for i in range(p):
            for j in range(p):
                ##Alternative method (for odd rank matrices)
                # if 0<=(i+l-b)<p and 0<=(j+k-b)<p:
                #     phi[l][k]+=(a[i][j])*(g[(i+l-b)][(j+k-b)])
                phi[l][k]+=(a[i][j])*(g[l-i][k-j])
    # norm=abs(g.sum())
    # if norm!=0:
    #     phi=phi/norm
    return phi

def convolve2(a):
    g=gen_decode(a)
    phi=np.zeros_like(a)
    p=np.size(a,1)
    a=np.fft.fft2(a,[p,p])
    g=np.fft.fft2(g,[p,p])
    phi=a*g
    phi=np.fft.ifft2(phi)
    #roll the matrix to match the Lecce images
    if p%2==0:
    # roll to get the Lecce convolution graph (even p)
        phi=np.roll(phi,[p//2-1,p//2-1],[1,0])
    else:
    #roll to get the Bologna convolution graph (odd p)
        phi=np.roll(phi,[-(p-1)//2,-(p-1)//2],[1,0])
    return phi



