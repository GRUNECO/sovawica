#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:29:12 2018

@author: jfochoa
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def wthresh(coeff,thr):
    y   = list()
    s = wnoisest(coeff)
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])))
    return y
    
def thselect(signal):
    Num_samples = 0
    for i in range(0,len(signal)):
        Num_samples = Num_samples + signal[i].shape[0]
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr

def wnoisest(coeff):
    stdc = np.zeros((len(coeff),1))
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745
    return stdc

mat_contents = sio.loadmat('/datos/instaladores/Matlab_2015/bin/senal_prueba_wavelet.mat')
data = np.squeeze(mat_contents['senal'])

LL = 8#int(np.floor(np.log2(data.shape[1])))

coeff = pywt.wavedec( data, 'db6', level=LL )

thr = thselect(coeff)
coeff_t = wthresh(coeff,thr)

x_rec = pywt.waverec( coeff_t, 'db6')

x_rec = x_rec[0:data.shape[0]]

plt.plot(x_rec[0:1500])
plt.plot(data[0:1500])

x_filt = np.squeeze(data - x_rec)
plt.plot(x_filt[0:1500])