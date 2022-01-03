import pywt
import numpy as np

def wthresh(X,thr):
    """
    Function to perform hard thresholding.

    Parameters:
        X: [cA_n, cD_n, cD_n-1, ..., cD2, cD1] list of 1-dimensional np.ndarrays
            Ordered list of coefficients arrays
            where ``n`` denotes the level of decomposition. The first element
            (``cA_n``) of the result is approximation coefficients array and the
            following elements (``cD_n`` - ``cD_1``) are details coefficients
            arrays.
        thr: float
                threshold value.
    
    Returns:
        list with hard T-thresholding of the input vector X.

    Inspired by:
        wthresh matlab wavelet function by M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    """
    y   = list()
    s = wnoisest(X)
    # 1 desde los detalles
    # 0 completo
    for i in range(0,len(X)):
        y.append(np.multiply(X[i],np.abs(X[i])>(thr*s[i])))
    return y
    
def thselect(X):
    """
    Threshold selection for de-noising.
    Rule: threshold is sqrt(2*log(length(X))) aka 'sqtwolog'

    Parameters:
        X: [cA_n, cD_n, cD_n-1, ..., cD2, cD1] list of 1-dimensional np.ndarrays
            Ordered list of coefficients arrays
            where ``n`` denotes the level of decomposition. The first element
            (``cA_n``) of the result is approximation coefficients array and the
            following elements (``cD_n`` - ``cD_1``) are details coefficients
            arrays.
    Returns:
        float-scalar with the threshold X-adapted value

    Notes:
        Threshold selection rule is based on the underlying
        model y = f(t) + e where e is a white noise N(0,1).

    Inspired by:
        thselect matlab wavelet function by M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    """
    Num_samples = 0
    for i in range(0,len(X)):
        Num_samples = Num_samples + X[i].shape[0]
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr

def wnoisest(coeff):
    """
    Estimates noise of 1-D wavelet coefficients.
    
    The estimator used is Median Absolute Deviation / 0.6745,
    well suited for zero mean Gaussian white noise in the 
    de-noising 1-D model.

    Parameters:
        coeff: [cA_n, cD_n, cD_n-1, ..., cD2, cD1] list of 1-dimensional np.ndarrays
            Ordered list of coefficients arrays
            where ``n`` denotes the level of decomposition. The first element
            (``cA_n``) of the result is approximation coefficients array and the
            following elements (``cD_n`` - ``cD_1``) are details coefficients
            arrays.

    Returns:
        numpy.ndarray with the noise estimates of each decomposition level

    Inspired by:
        wnoisest matlab wavelet function by M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    """
    stdc = np.zeros((len(coeff),1))
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745
    return stdc

def W_ICA_epoch(signal,axis=-1): 
    """
    Applies wavelet filtering to continuous data (ie a single epoch).

    Parameters:
        signal: numpy.ndarray
            ica components in the shape of (sources,samples)
        axis: integer, default: -1 (last axis)
            axis corresponding to the samples of the component
    Returns:
        numpy.ndarray with the filtered signal

    Inspired by:
        wnoisest matlab wavelet function by M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    """
    LL = int(np.floor(np.log2(len(signal)))) # define the number of levels
    if LL>6:
      LL=6
    #print(LL) # giving level 11 currently
    # Wavelet decomposition of x.
    coeff = pywt.wavedec(signal, 'db6', level=LL,axis=axis) #was fixed to level = 8
    thr = thselect(coeff)
    coeff_t = wthresh(coeff,thr)
    # Wavelet reconstruction of xd.
    x_rec = pywt.waverec(coeff_t, 'db6',axis=axis)
    x_rec = x_rec[0:signal.shape[0]]
    x_filt = np.squeeze(signal - x_rec)
    return x_filt

def removeStrongArtifacts_wden(ica_segmented,components,k=4,mode='sigma'):
    """
        Removes strong artifacts of epoched-data using wavelet filtering.

        Parameters:
            ica_segmented: numpy.ndarray
                ica components in the shape of (sources,samples,epochs)
            components: iterable object 
                list/1D-array of components to filter numbered from 0 and as integers
            k : float, default: 4
                Tolerance for cleaning artifacts (ie 1,1.15,4)
            mode: 'sigma'|'iqr'
        Returns:
            tuple of numpy.ndarray objects
                filtered signal , vector of filtered components (1|YES,0|NO)
    """
    (sources,samples,epochs) = ica_segmented.shape
    ica_return = np.copy(ica_segmented)
    opt = np.zeros((ica_segmented.shape[0],epochs))
    
    #print('Wavelet Component Filtering:' + str(len(components)))
    for comp in components:
        # Slice to component
        Y = ica_segmented[comp,:,:] # obtenemos los datos la componente
        Y = np.reshape(Y,(1,samples*epochs),order='F') # pasamos esto a un vector continuo
        
        # select noisy epochs for cleaning given the component
        sig = np.median(np.divide(np.abs(Y),0.6745)) # determinamos sigma
        q3, q1 = np.percentile(np.abs(Y), [75 ,25]) # percentiles
        iqr = q3 - q1 # rango intercuartil
        #k=iqr
        if mode == 'sigma':
          thr = k*sig #1*sig#4*sig # umbral de corte
        elif mode == 'iqr':
          thr = q3+1.5*iqr
        idx = np.where(np.abs(Y)>thr) # idx = muestras (indices) donde se incumple el umbral
        if (idx[0].shape[0]==0):
            #disp(['The component #' num2str(Comp(c)) ' has passed unchaneged']);
            print('The component' + str(comp) ,'has passed unchanged')
            continue 
        else:
            #$if the component is noisy try to detect the bad epoch and
            #%reconstruct
            for epoch in range(epochs):
                Y = ica_segmented[comp,:,epoch] # shape= todos los puntos de una sola epoca
                idx = np.where(np.abs(Y)>thr) # idx = punto de la epoca donde el umbral se incumple
                if (idx[0].shape[0]==0):
                    continue 
                else:
                    # Filter wavelet
                    xn =  W_ICA_epoch(Y)
                    #print(np.sum(xn - Y))
                    opt[comp,epoch] = 1
                    ica_return[comp,:,epoch] = xn
                    '''
                    plt.subplot(3,1,1)
                    plt.plot(xn)
                    plt.subplot(3,1,2)
                    plt.plot(Y)
                    plt.subplot(3,1,3)
                    plt.plot(xn - Y)
                    plt.show()
                    '''
                    # the print below actually makes the algorithm a lot slower
                    #print('The component #' + str(comp) + ' has been filtered in the epoch #' + str(epoch))
        #print('The component #' + str(comp) + ' has been filtered')
        #print(str((comp+1)*100/len(components)) + '%', end = ' ')
        #print(comp,end=' ')
    #print('WICA DONE')
    return ica_return,opt

def w_ica_matlab(ica_segmented,components,k=4,mode='sigma'):
  """
  Wrapper function of removeStrongArtifacts_wden.
  """
  return removeStrongArtifacts_wden(ica_segmented,components,k,mode)
