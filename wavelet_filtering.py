#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:29:12 2018

@author: jfochoa
"""

import pywt
import numpy as np
#import seaborn
from statsmodels.robust import mad
import matplotlib.pyplot as plt
import scipy.io as sio
from mne.io.cnt import read_raw_cnt

"""
Implementation of the PREP pipeline in python.
"""
#%% Start
# Path hack
import sys, os
sys.path.insert(0, os.path.abspath('...'))

import numpy as np
import mne
from mne.io.cnt import read_raw_cnt
from mne.io import read_raw_eeglab
from pytictoc import TicToc


import prepy.pyfirfilt as pff
import prepy.utils as utl
import prepy.reference as ref
import prepy.detrend as dt
import scipy.io as sio
import csv

"""
PREP params:
   params.referenceChannels = 1:nb_channels;
   params.evaluationChannels = 1:nb_channels;
   params.rereferencedChannels = 1:nb_channels;
   params.detrendChannels = 1:nb_channels;
   params.lineNoiseChannels = 1:nb_channels;
   params.lineFrequencies = [60, 120, 180, 212, 240];
   params.detrendType = 'high pass';
   params.detrendCutoff = 1;
   params.referenceType = 'robust';
   params.keepFiltered = false;
"""

#%% Data Loading
# Preload for faster indexing and manipulation

#file_name = 'CARLOS-N400'
file_name = 'CTR_009_V1_N4'
data_path = 'datasets/'
output = 'output/'
fullpath = 'C:/Users/user/Desktop/code/preprocessing-flow/'


raw = read_raw_cnt(data_path + file_name+'.cnt', preload=True, stim_channel=False)
raw.resample(300, npad='auto', verbose='error') # To not get out of memory on RANSAC

#%% Getting rid of ocular channels for EEG preprocessing
#Correct the type of ocular channels (Needed because standard montage doesnt have them)

raw.set_channel_types(mapping={'HEO': 'eog'})
raw.set_channel_types(mapping={'VEO': 'eog'})
raw.pick_types(meg=False,eeg=True,eog=False)

#%% Montage selection
# If using standard montage
montage = mne.channels.make_standard_montage('standard_1020')

# convert mne naming system to our system (mne system is not ALLCAPS)
montage.ch_names = [x.upper() for x in montage.ch_names]
raw.set_montage(montage, verbose=False)

#%% Preparing data scale (matlab is not in V, but in uV)
MATLAB_SCALE_FACTOR = 10 ** (-6)
PYTHON_SCALE_FACTOR = 1/MATLAB_SCALE_FACTOR
 
#def Threshold(Signal, Type_est = 1):
#    """Calcula el valor del umbral fijo adecuado para una señal, utilizando alguno de los estimadores estadísticos.
#
#    Parámetros:
#        Signal: tipo numpy.ndarray
#                Vector con los datos de la señal. 
#            
#        Type_est: tipo int
#                Valor que indica el tipo de estimadores estadísticos a emplear:
#                Type_est = 1 --> 'rigrsure'
#                Type_est = 2 --> 'heursure'
#                Type_est = 3 --> 'sqtwolog'
#                Type_est = 4 --> 'minimaxi'
#
#    Devuelve:
#        Thr: tipo float
#                Valor del umbral fijo calculado por el tipo de estimador estadístico especificado.
#                Si el valor retornado es 0, puede significar que se ingreso un tipo de estimador estadístico
#                incorrecto.
#    """
#    # Sacar el numero de datos de la señal
#    Num_samples = Signal.size 
#    # Verificar el tipo de estimador estadistico a utilizar
#    if Type_est == 1:   # Si es de tipo 'rigrsure'
#        sx2 = (np.sort(np.absolute(Signal)))**2
#        risks = ((Num_samples-(2*(np.arange(1,Num_samples+1))))+((np.cumsum(sx2))+((np.arange(Num_samples-1,-1,-1))*sx2)))/Num_samples
#        best = np.argmin(risks)
#        thr = np.sqrt(sx2[best])
#    elif Type_est == 2: # Si es de tipo 'heursure'
#        hthr = np.sqrt(2*(np.log(Num_samples)))
#        eta = ((np.linalg.norm(Signal)**2)-Num_samples)/Num_samples
#        crit = (((np.log(Num_samples))/(np.log(2)))**(1.5))/(np.sqrt(Num_samples))
#        if eta < crit:
#            thr = hthr
#        else:
#            thr = np.amin([Threshold(Signal, 1),hthr])
#    elif Type_est == 3: # Si es de tipo 'sqtwolog'
#        thr = np.sqrt(2*(np.log(Num_samples)))
#    elif Type_est == 4: # Si es de tipo 'minimaxi'
#        if Num_samples <= 32:
#            thr = 0
#        else:
#            thr = 0.3936 + 0.1829*((np.log(Num_samples))/(np.log(2)))
#    else:
#        # En el caso que se ingrese un valor distinto de 1, 2, 3 y 4, no es ningun tipo definido
#        print('\n\n\tEl tipo de estimador estadistico del umbral no esta definido!!!.')
#        # Se devuelve un 0 ya que no es un tipo definido de estimador estadístico 
#        return 0
#    # Se devuelve el valor del umbral calculado 
#    return float(thr)
#
#
#def NoiseLevel(Signal):
#    """Calcula el nivel de ruido de una señal, usando el método de “median absolute deviation”.
#
#    Parámetros:
#        Signal: tipo numpy.ndarray
#                Vector con los datos de la señal.
#
#    Devuelve:
#        Level: tipo float
#                Valor del nivel de ruido de la señal.
#    """
#    # Calcular el nivel de ruido de la señal 
#    Level = (np.median(np.absolute(Signal)))/0.6745
#    # Se devuelve el valor del nivel de ruido calculado 
#    return float(Level)
#
#
#def Thresholding(Signal, Threshold, Type_def = 1):
#    """Realiza la funcion de umbralizacion de tipo hard o soft, apartir de un valor de umbral. 
#
#    Parámetros:
#        Signal: tipo numpy.ndarray
#                vector con los datos de la señal. 
#        
#        Threshold: tipo float
#                Valor del umbral fijo para ser usado en la umbralizacion.
#            
#        Type_def: tipo int
#                Valor que indica el tipo de umbralizacion a emplear:
#                Type_def = 1 --> 'hard'
#                Type_def = 2 --> 'soft'
#
#    Devuelve:
#        New_signal: tipo numpy.ndarray
#                Vector con los nuevos datos de la señal (señal umbralizada).
#                Si el valor retornado es 0, significa que se ingreso un tipo de umbralizacion
#                incorrecta.
#    """
#    # crear el vector que va a contener la nueva señal
#    New_signal = np.zeros(Signal.shape[0])
#    # Verificar el tipo de umbralizacion a realizar
#    if Type_def == 1:   # Si es de tipo hard
#        # Evaluar el valor absoluto de cada dato de la señal con el umbral fijo y si es menor se reemplaza por 0
#        for i in range(0,New_signal.shape[0]):
#            if np.absolute(Signal[i]) < Threshold: 
#                New_signal[i] = 0
#            else:   # Mantener en caso de que el valor se encuentre dentro del rango
#                New_signal[i] = Signal[i]
#    elif Type_def == 2: # Si es de tipo soft
#        for i in range(0,New_signal.shape[0]):
#            if np.absolute(Signal[i]) < Threshold: # Evaluar el valor absoluto de cada dato de la señal con el umbral fijo y si es menor se reemplaza por 0
#                New_signal[i] = 0
#            elif Signal[i] >= Threshold: # Evaluar cada dato de la señal con el umbral fijo y si es mayor se le resta el umbral
#                New_signal[i] = Signal[i] - Threshold
#            elif Signal[i] <= -1*Threshold: # Evaluar cada dato de la señal con el umbral fijo negativo y si es menor se le suma el umbral
#                New_signal[i] = Signal[i] + Threshold
#    else:
#        # En el caso que se ingrese un valor distinto de 1 y 2 no es ningun tipo definido
#        print('\n\n\tEl tipo de umbralizacion no esta definido!!!.')
#        # Se devuelve un 0 ya que no es un tipo definido de umbralizacion 
#        return 0
#    # Se devuelve la nueva señal umbralizada
#    return New_signal


def wthresh(coeff,thr):
    y   = list()
    s = wnoisest(coeff)
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])))
    return y
    
def thselect(signal, Type_est = 1):
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
#thr = thselect(c,tptr)
thr = thselect(coeff,3)

coeff_t = wthresh(coeff,thr)

x_rec = pywt.waverec( coeff_t, 'db6')

x_rec = x_rec[0:data.shape[0]]

plt.plot(x_rec[0:1000])
plt.plot(data[0:1000])

x_filt = np.squeeze(data - x_rec)
plt.plot(x_filt[0:1000])

## %%
#
#LL = floor(log2(length(Y)))
#
#xd = wden(Y,'sqtwolog','h','one',LL,'db6')
#
#
#[c,l] = wavedec(Y,LL,db6)
#
#% Threshold rescaling coefficients.
#switch scal
#  case 'one' , s = ones(1,n)
#  case 'sln' , s = ones(1,n)*wnoisest(c,l,1)
#  case 'mln' , s = wnoisest(c,l,1:n)
#    otherwise
#        error(message('Wavelet:FunctionArgVal:Invalid_ArgVal'))
#end
#
#% Wavelet coefficients thresholding.(l tiene los rangos para los )
#first = cumsum(l)+1
#first = first(end-2:-1:1)
#ld   = l(end-1:-1:2)
#last = first+ld-1
#cxd = c
#lxd = l
#for k = 1:n
#    flk = first(k):last(k)
#    if strcmp(tptr,'sqtwolog') || strcmp(tptr,'minimaxi')
#        thr = thselect(c,tptr)
#    else
#        if s(k) < sqrt(eps) * max(c(flk))
#            thr = 0
#        else
#            thr = thselect(c(flk)/s(k),tptr)
#        end
#    end                                     % threshold.
#    thr      = thr * s(k)                  % rescaled threshold.
#    cxd(flk) = wthresh(c(flk),sorh,thr)    % thresholding or shrinking.
#end
#
#% Wavelet reconstruction of xd.
#xd = waverec(cxd,lxd,w) 