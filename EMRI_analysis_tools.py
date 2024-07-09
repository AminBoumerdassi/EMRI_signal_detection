"""
Functions for analysis EMRIs after they have been generated
"""

import numpy as np
from lisatools.sensitivity import *


def inner_prod_AET(sig1_t_AET, sig2_t_AET, delta_t, PSD, use_gpu=True):#N_t,
    """ Vectorised inner product for input shape (no. chans, length timeseries)
    
    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """

    if use_gpu:#Fine to keep this; these variables are local
        xp=cp
    else:
        xp=np
    
    N_t= sig1_t_AET.shape[1]#len(sig1_t)

    #FFT the two signals
    freq= xp.fft.rfftfreq(N_t, delta_t)
    freq[0] = freq[1]   # To "retain" the zeroth frequency

    sig1_f= xp.fft.rfft(sig1_t_AET)
    sig2_f_conj= xp.fft.rfft(sig2_t_AET).conj()

    #Calculate the PSD
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in range(sig1_t_AET.shape[0])])
    #PSD_AET= get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD")


    #Calculate the prefactor
    prefac = 4*delta_t / N_t

    #Calculate the output inn. prod.
    out= prefac* xp.real(xp.sum((sig1_f*sig2_f_conj)/PSD_AET, axis=1))

    return out

def inner_prod_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, PSD, use_gpu=True):#N_t,
    """ Vectorised inner product for input shape (batch size, no. chans, length timeseries)
    
    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """

    if use_gpu:#Fine to keep this; these variables are local
        xp=cp
    else:
        xp=np
    
    N_t= sig1_t_AET.shape[2]#len(sig1_t)

    #FFT the two signals
    freq= xp.fft.rfftfreq(N_t, delta_t)
    freq[0] = freq[1]   # To "retain" the zeroth frequency

    sig1_f= xp.fft.rfft(sig1_t_AET)
    sig2_f_conj= xp.fft.rfft(sig2_t_AET).conj()

    #Calculate the PSD
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in range(sig1_t_AET.shape[1])])
    #PSD_AET= get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD")


    #Calculate the prefactor
    prefac = 4*delta_t / N_t

    #Calculate the output inn. prod.
    out= prefac* xp.real(xp.sum((sig1_f*sig2_f_conj)/PSD_AET, axis=2))

    return out



    '''Overlap across AET is:
    Overlap_AET= sqrt[(1/n_chans) * sum_across_chans(Overlap_chan^2)]
    '''

def overlap_AET(sig1_t_AET, sig2_t_AET, delta_t, PSD, use_gpu=True):
    """ Network overlap for input shape (no. chans, length timeseries)

    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """

    if use_gpu:#Fine to keep this; these variables are local
        xp=cp
    else:
        xp=np

    inner_prod_h1_h2= inner_prod_AET(sig1_t_AET, sig2_t_AET, delta_t, None, use_gpu=use_gpu)

    inner_prod_h1_h1= inner_prod_AET(sig1_t_AET, sig1_t_AET, delta_t, None, use_gpu=use_gpu)

    inner_prod_h2_h2= inner_prod_AET(sig2_t_AET, sig2_t_AET, delta_t, None, use_gpu=use_gpu)

    overlap= (inner_prod_h1_h2)/np.sqrt(inner_prod_h1_h1* inner_prod_h2_h2)

    return np.sqrt(np.sum(overlap**2)/sig1_t_AET.shape[0])

def overlap_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, PSD, use_gpu=True):
    """ Network overlap for input shape (batch size, no. chans, length timeseries)

    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """

    if use_gpu:#Fine to keep this; these variables are local
        xp=cp
    else:
        xp=np

    inner_prod_h1_h2= inner_prod_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, None, use_gpu=use_gpu)

    inner_prod_h1_h1= inner_prod_AET_batchwise(sig1_t_AET, sig1_t_AET, delta_t, None, use_gpu=use_gpu)

    inner_prod_h2_h2= inner_prod_AET_batchwise(sig2_t_AET, sig2_t_AET, delta_t, None, use_gpu=use_gpu)

    overlap= (inner_prod_h1_h2)/np.sqrt(inner_prod_h1_h1* inner_prod_h2_h2)

    return np.sqrt(np.sum(overlap**2, axis=1)/sig1_t_AET.shape[1])

