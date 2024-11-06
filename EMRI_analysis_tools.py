"""
Functions for analysis EMRIs after they have been generated
"""

import numpy as np
import cupy as cp
from lisatools.sensitivity import *
from scipy.signal.windows import tukey

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import SchwarzschildEccentricWaveformBase,FastSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.constants import YRSID_SI

#LISA tools imports
from lisatools.sensitivity import *

#fast lisa response imports
from fastlisaresponse import ResponseWrapper

def zero_pad(data):#, xp=xp
    """
    Zero pads an array so its length is a power of two.
    """
    #Check whether to use cupy or numpy
    xp = cp.get_array_module(data)

    N = len(data)
    pow_2 = xp.ceil(xp.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def zero_pad_BATCHWISE(data):
    """
    Zero-pads a batch of vectors to length 2^x.
    Input: (batch_size, no. channels, vector_length)
    Output: (batch_size, no. channels, padded_vec_length)
    """
    #Check whether to use cupy or numpy
    xp = cp.get_array_module(data)

    N = data.shape[2]#len(data)
    pow_2 = xp.ceil(xp.log2(N))
    pad_width= ((0,0),(0,0),(0,int((2**pow_2)-N)))
    return xp.pad(data, pad_width, 'constant')
    
def noise_td_AET(N, dt, channels=["AE","AE","T"], return_cupy=True):
    """ 
    Generate time-domain noise coloured by the AET channel PSDs.
    """
    #Check whether to use cupy or numpy
    if return_cupy==True:
        xp=cp
    else:
        xp=np

    #Extract frequency bins for use in PSD
    N_padded= len(zero_pad(xp.ones(N)))
    freq = xp.fft.rfftfreq(N_padded , dt)
    freq[0] = freq[1]#avoids NaNs in PSD[0]
    
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in channels])

    #Draw samples from multivariate Gaussian
    variance_noise_f= 0.25* N_padded*dt*PSD_AET#N*PSD_AET/(4*dt)#should we use N_padded rather than N?
    noise_f = xp.random.normal(0,xp.sqrt(variance_noise_f)) + 1j*xp.random.normal(0,xp.sqrt(variance_noise_f))
    #Transforming the frequency domain signal into the time domain
    return xp.fft.irfft(noise_f)[:,:N]

def noise_td_AET_BATCHWISE(N, dt, batch_size, channels=["AE","AE"], return_cupy=True):
    """
    Generate batches of TD AET noise.
    output: (batch_size, no. channels, time_steps) 
    """
    #Check whether to use cupy or numpy
    if return_cupy==True:
        xp=cp
    else:
        xp=np

    #Pad N to nearest power of 2 for faster FFT calculation
    N_padded= len(zero_pad(xp.ones(N)))

    #Extract frequency bins for use in PSD
    freq = xp.fft.rfftfreq(N_padded, dt)
    freq[0] = freq[1]#avoids NaNs in PSD[0]

    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in channels])
    PSD_AET_BATCHWISE= PSD_AET * xp.ones((batch_size, len(channels), len(PSD_AET[0])))

    #Draw samples from multivariate Gaussian
    variance_noise_f= 0.25* N_padded*dt*PSD_AET_BATCHWISE#N*PSD_AET_BATCHWISE/(4*dt)#should we use N_padded rather than N?
    noise_f = xp.random.normal(0,xp.sqrt(variance_noise_f)) + 1j*xp.random.normal(0,xp.sqrt(variance_noise_f))
    #Transforming the frequency domain signal into the time domain
    return xp.fft.irfft(noise_f)[:,:,:N]

def init_TDI(dim, dt, TDI_channels=['TDIA','TDIE','TDIT'], use_gpu=True):
        # self.dim = dim
        # self.dt = dt
        # self.TDI_channels=TDI_channels
        T= (dim*dt/YRSID_SI)+0.005#A tiny bit extra on T to ensure output length =>dim
        # self.channels_dict= {"AET":["AE","AE","T"], "AE":["AE","AE"]}#For use in the noise generation and whitening functions
        # self.n_channels = len(TDI_channels)
        #self.shuffle = shuffle
        # self.add_noise= add_noise
        #self.on_epoch_end()
        # self.seed= seed
        
        # #initialise RNG for noise generation with a fixed seed
        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed_all(self.seed)
        # np.random.seed(self.seed)
        # xp.random.seed(self.seed)

        # order of the langrangian interpolation
        t0 = 20000.0   # How many samples to remove from start and end of simulations
        order = 25
        orbit_file_esa = "/../../../../fred/oz303/aboumerd/software/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"#"/nesi/project/uoa00195/software/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
        orbit_kwargs_esa = dict(orbit_file=orbit_file_esa) # these are the orbit files that you will have cloned if you are using Michaels code.
        # you do not need to generate them yourself. They’re already generated. 

        # 1st or 2nd or custom (see docs for custom)
        tdi_gen = "2nd generation"
        tdi_kwargs_esa = dict(
            orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan=TDI_channels)#['TDIA','TDIE','TDIT'], ["TDI"+i for i in TDI_channels], , num_pts=self.dim

        #Specify the indices of the sky coordinates in the array of parameters
        index_lambda = 7 # Index of polar angle
        index_beta = 8   # Index of phi angle

        #Kwargs for the waveform generator
        waveform_kwargs={"sum_kwargs":{"pad_output":True}}

        #Initialise the waveform generator
        generic_class_waveform_0PA_ecc = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux", use_gpu = use_gpu, **waveform_kwargs)
        #Then initialise the response wrapper
        return ResponseWrapper(generic_class_waveform_0PA_ecc, T, dt,
                                        index_lambda, index_beta, t0=t0,
                                        flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                                        remove_garbage = True,  **tdi_kwargs_esa)


def generate_TDI_EMRI(EMRI_params, TDI_wrapper):
    '''
    Generate ONE EMRI using the initialised TDI/response wrapper
    from a set of parameters.
    '''
    return TDI_wrapper(*EMRI_params)
    
def get_TDI_noise(batch_size, dt, n_chans, len_arr, return_cupy=True):
    '''
    Generate ONE batch of TDI LISA noise. Not for overlaying on GW events
    since this is already whitened! More useful for tests involving pure noise.

    Output shape: (batch_size, dim, n_channels)
    '''
    if return_cupy==True:
        xp=cp
    else:
        xp=np
    
    #Define the output array
    batch_TDI_noise= xp.empty((batch_size, n_chans, len_arr))

    #Iterate noise generation and whitening over one batch
    batch_TDI_noise= noise_td_AET_BATCHWISE(len_arr, dt, batch_size, channels=["AE","AE"][:n_chans+1], return_cupy=return_cupy)
    batch_TDI_noise= noise_whiten_AET_BATCHWISE(batch_TDI_noise)

    # for i in range(batch_size):
    #     noise_AET= noise_td_AET(len_arr, dt, channels=self.channels_dict[self.TDI_channels])
    #     #Then whiten
    #     batch_TDI_noise[i,:,:]= noise_whiten_AET(noise_AET, dt, channels=self.channels_dict[self.TDI_channels])
    #Reshape to have the correct shape for the model
    # batch_TDI_noise= xp.swapaxes(batch_TDI_noise, 1, 2).copy()
    #batch_TDI_noise= xp.reshape(batch_TDI_noise, (self.batch_size, self.dim, self.n_channels))
    return batch_TDI_noise

def noise_whiten_AET(noisy_signal_td_AET, dt, channels=["AE","AE","T"]):
    '''This is vectorised for the AET channels.        
        May be quicker or more convenient if we use PyTorch's FFT and windowing. Worth testing

        ISSUE: this function returns weird results for >5000000 datapoints. Check closely.

        '''
    #Check whether to use cupy or numpy
    xp = cp.get_array_module(noisy_signal_td_AET)

    #FFT the windowed TD signal; obtain freq bins
    signal_length= len(noisy_signal_td_AET[0])
    window= xp.asarray(tukey(signal_length, alpha=0))# alpha=0,1/8
    padded_noisy_signal_td_AET= xp.asarray([zero_pad(window*noisy_signal_td) for noisy_signal_td in noisy_signal_td_AET])
    
    noisy_signal_fd_AET= xp.fft.rfft(padded_noisy_signal_td_AET)
    
    padded_signal_length= len(padded_noisy_signal_td_AET[0])
    freq = xp.fft.rfftfreq(padded_signal_length, dt)
    freq[0]=freq[1]#To avoid NaN in PSD[0]
    
    #Divide FD signal by ASD of noise
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in channels])#"noisepsd_"+channel
            
    scaling_factor= np.sqrt(2.0/(PSD_AET*dt))#((PSD_AET)/(2*dt))**-0.5#len(noisy_signal_td)
    whitened_signal_fd_AET= scaling_factor*noisy_signal_fd_AET

    #IRFFT and truncate the junk at the end
    whitened_signal_td_AET= xp.fft.irfft(whitened_signal_fd_AET, n=padded_signal_length)[:,:signal_length]

    #IFFTing back into the time domain
    return whitened_signal_td_AET

def noise_whiten_AET_BATCHWISE(noisy_signal_td_AET, dt, batch_size, channels=["AE","AE","T"]):
    '''
    Noise-whiten batchwise across the AET channels.
    It may also be quicker if we use PyTorch's FFT and windowing. Worth testing
    '''
    #Check whether to use cupy or numpy
    xp = cp.get_array_module(noisy_signal_td_AET)

    #FFT the windowed TD signal; obtain freq bins
    signal_length= noisy_signal_td_AET.shape[-1]
    window= xp.asarray(tukey(signal_length, alpha=0))#*xp.ones((batch_size, len(channels), signal_length))

    padded_noisy_signal_td_AET= zero_pad_BATCHWISE(window*noisy_signal_td_AET)#xp.asarray([zero_pad_BATCHWISE(window*noisy_signal_td) for noisy_signal_td in noisy_signal_td_AET])
    
    noisy_signal_fd_AET= xp.fft.rfft(padded_noisy_signal_td_AET)
    
    signal_length_padded= padded_noisy_signal_td_AET.shape[-1]
    freq = xp.fft.rfftfreq(signal_length_padded, dt)
    freq[0]=freq[1]#To avoid NaN in PSD[0]
    
    #Divide FD signal by ASD of noise
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in channels])#"noisepsd_"+channel
    
    PSD_AET_BATCHWISE= PSD_AET# * xp.ones((batch_size, len(channels), len(PSD_AET[0])))

    scaling_factor= np.sqrt(2.0/(PSD_AET_BATCHWISE*dt))#((PSD_AET_BATCHWISE)/(2*dt))**-0.5#len(noisy_signal_td)
    whitened_signal_fd_AET= scaling_factor*noisy_signal_fd_AET

    #IFFTing back into the time domain; truncate the zero-padding
    whitened_signal_td_AET= xp.fft.irfft(whitened_signal_fd_AET, n=signal_length_padded)[:,:,:signal_length]

    return whitened_signal_td_AET

###
#other functions used after training to evaluate model accuracy and generalisability
###

def inner_prod(sig1_t,sig2_t,N_t,delta_t,PSD, use_gpu=True):
    """ This is only valid if:
        1. signals are same length
        2. signals have same no. of channels
    """

    if use_gpu:#Fine to keep this; these variables are local
        xp=cp
    else:
        xp=np
    
    #FFT the two signals
    freq= xp.fft.fftfreq(N_t, delta_t)
    freq[0] = freq[1]   # To "retain" the zeroth frequency

    sig1_f= xp.fft.rfft(sig1_t)
    sig2_f_conj= xp.fft.rfft(sig2_t).conj()

    #Calculate the PSD
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in sig1_t.shape[0]])
    
    #Calculate the prefactor
    prefac = 4*delta_t / N_t

    #Calculate the output inn. prod.
    out= prefac* xp.real(xp.sum((sig1_f*sig2_f_conj)/PSD_AET))

    return out


def inner_prod_AET(sig1_t_AET, sig2_t_AET, delta_t, PSD, use_gpu=True):
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
    Formula taken from p9 of https://arxiv.org/pdf/2310.08927

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

    overlap= (inner_prod_h1_h2)/xp.sqrt(inner_prod_h1_h1* inner_prod_h2_h2)

    return xp.sqrt(xp.sum(overlap**2)/sig1_t_AET.shape[0])

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

    overlap= (inner_prod_h1_h2)/xp.sqrt(inner_prod_h1_h1* inner_prod_h2_h2)

    return xp.sqrt(xp.sum(overlap**2, axis=1)/sig1_t_AET.shape[1])

