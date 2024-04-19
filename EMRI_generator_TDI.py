'''
This is a custom TensorFlow data generator object for generating time-domain EMRIs from a given set of parameters.
It uses sets of EMRI parameters to generate and store time-domain EMRIs only for as long as is needed in a particular batch. 
'''
#---------------------------------------------------------------------------------------
#Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#---------------------------------------------------------------------------------------

#GPU check
use_gpu = True

import numpy as np
import cupy as xp
from tensorflow import keras
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack


#FEW imports
import sys
import os
from numpy.random import default_rng
from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import SchwarzschildEccentricWaveformBase,FastSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.constants import YRSID_SI

#LISA tools imports
from lisatools.sensitivity import *

#fast lisa response imports
from fastlisaresponse import ResponseWrapper

#oise whitening/noise generation imports
from scipy.signal.windows import tukey


class EMRIGeneratorTDI(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, EMRI_params_dir, batch_size=32, dim=2**21, dt=10.,  TDI_channels="AET",
                  shuffle=True, seed=2023, add_noise=True):#list_IDs, T=1.,  TDI_channels=['TDIA','TDIE','TDIT']
        'Initialization'
        #self.T = T
        #self.list_IDs= list_IDs
        self.EMRI_params_dir = EMRI_params_dir

        """Loading all parameter sets is not ideal since for a very large dataset e.g. 500K EMRIs,
            this would eat into the available memory. For reference, if self.EMRI_params had dimensions
            (500K,14) it would take up ???GB."""
        self.EMRI_params= np.load(self.EMRI_params_dir, allow_pickle=True)
        self.EMRI_params_set_size= self.EMRI_params.shape[0]

        self.batch_size = batch_size
        self.dim = dim
        self.dt = dt
        self.TDI_channels=TDI_channels
        self.T= dim*dt/YRSID_SI
        self.channels_dict= {"AET":["AE","AE","T"], "AE":["AE","AE"]}#For use in the noise generation and whitening functions
        self.n_channels = len(TDI_channels)
        self.shuffle = shuffle
        self.add_noise= add_noise
        self.on_epoch_end()
        self.seed= seed
        
        #initialise RNG for noise generation with a fixed seed
        np.random.seed(seed=self.seed)
        
        #Initialise the TDI/response wrapper
        # order of the langrangian interpolation
        t0 = 20000.0   # How many samples to remove from start and end of simulations
        order = 25
        orbit_file_esa = "/nesi/project/uoa00195/software/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
        orbit_kwargs_esa = dict(orbit_file=orbit_file_esa) # these are the orbit files that you will have cloned if you are using Michaels code.
        # you do not need to generate them yourself. They’re already generated. 

        # 1st or 2nd or custom (see docs for custom)
        tdi_gen = "2nd generation"
        tdi_kwargs_esa = dict(
            orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan=self.TDI_channels)#['TDIA','TDIE','TDIT'], ["TDI"+i for i in TDI_channels], , num_pts=self.dim

        #Specify the indices of the sky coordinates in the array of parameters
        index_lambda = 7 # Index of polar angle
        index_beta = 8   # Index of phi angle

        #Kwargs for the waveform generator
        waveform_kwargs={"sum_kwargs":{"pad_output":True}}

        #Initialise the waveform generator
        generic_class_waveform_0PA_ecc = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux", use_gpu = use_gpu, **waveform_kwargs)
        #Then initialise the response wrapper
        self.EMRI_TDI_0PA_ecc = ResponseWrapper(generic_class_waveform_0PA_ecc, self.T, self.dt,
                                        index_lambda, index_beta, t0=t0,
                                        flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                                        remove_garbage = "zero", n_overide= self.dim, **tdi_kwargs_esa)

    def __len__(self):
        'Denotes the number of batches per epoch'
        '''
        Ideally this should be int(np.floor(self.EMRI_params_set_size / self.batch_size))
        
        But this leads to too many steps which makes training too long.
        
        So for now, keep this at some small number.

        Later on, when we have a working model, 
        '''
        return 10#1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        #list_IDs_temp = [self.dataset_len[k] for k in indexes]

        
        X, y = self.__data_generation(indexes)#list_IDs_temp
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.EMRI_params_set_size)#len(self.list_IDs)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            #self.rng.shuffle(self.indexes)
            

    def __data_generation(self, temp_indexes):#list_IDs_temp
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialise an empty batch of training data
        X = xp.empty((self.batch_size, self.n_channels, self.dim))
                
        # Iterate EMRI waveform generation for our batch
        for i, batch_index in zip(temp_indexes, np.arange(self.batch_size)):#list_IDs_temp            
            waveform= self.generate_TDI_EMRI(self.EMRI_params[i,:])
            
            #Then preprocess with noise and whitening
            noise_AET= self.add_noise * self.noise_td_AET(self.dim, self.dt, channels=self.channels_dict[self.TDI_channels])#["AE","AE","T"]
            noisy_signal_AET= xp.asarray(waveform)+noise_AET
            X[batch_index,:,:]= self.noise_whiten_AET(noisy_signal_AET, self.dt, channels=self.channels_dict[self.TDI_channels])

        #Standardising inputs to have mean 0, variance 1
        mu= xp.mean(X, axis=2).reshape(X.shape[0],X.shape[1],1)
        stdev= xp.std(X, axis=2).reshape(X.shape[0],X.shape[1],1)

        X= (X-mu)/stdev

        #Reshape X and copy X for the model
        X=  xp.swapaxes(X, 1, 2).copy()#xp.reshape(X, (self.batch_size, self.dim, self.n_channels))#.get()


        #Convert X from xp arrays to TF tensors
        X= self.cupy_to_tensor(X)
        return X, X      

        
    def zero_pad(self, data):
        """
        This function takes in a vector and zero pads it so it is a power of two.
        """
        N = len(data)
        pow_2 = xp.ceil(cp.log2(N))
        return xp.pad(data,(0,int((2**pow_2)-N)),'constant')
        
    def noise_td_AET(self, N, dt, channels=["AE","AE","T"]):
        """ 
        This is vectorised for the AET channels!
        GPU-enabled only!
        """
        #Extract frequency bins for use in PSD
        N_padded= len(self.zero_pad(xp.ones(N)))
        freq = xp.fft.rfftfreq(N_padded , dt)
        freq[0] = freq[1]#avoids NaNs in PSD[0]
        
        PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn="noisepsd_"+channel, return_type="PSD") for channel in channels])

        #Draw samples from multivariate Gaussian
        variance_noise_f= N*PSD_AET/(4*dt)
        noise_f = xp.random.normal(0,np.sqrt(variance_noise_f)) + 1j*xp.random.normal(0,np.sqrt(variance_noise_f))
        #Transforming the frequency domain signal into the time domain
        return xp.fft.irfft(noise_f, n=N)
    
    def generate_TDI_EMRI(self, EMRI_params):#response_wrapper,
        '''
        Generate ONE EMRI using the initialised TDI/response wrapper
        from a set of parameters.
        '''
        return self.EMRI_TDI_0PA_ecc(*EMRI_params)#response_wrapper
        
    def get_TDI_noise(self):
        '''
        Generate ONE batch of TDI LISA noise. Not for overlaying on GW events
        since this is already whitened! More useful for tests involving pure noise.

        Output shape: (batch_size, dim, n_channels)
        '''
        #Define the output array
        batch_TDI_noise= xp.empty((self.batch_size, self.n_channels, self.dim))

        #Iterate noise generation and whitening over one batch
        for i in range(self.batch_size):
            noise_AET= self.noise_td_AET(self.dim, self.dt, channels=self.channels_dict[self.TDI_channels])
            #Then whiten
            batch_TDI_noise[i,:,:]= self.noise_whiten_AET(noise_AET, self.dt, channels=self.channels_dict[self.TDI_channels])
        #Reshape to have the correct shape for the model
        batch_TDI_noise= xp.swapaxes(batch_TDI_noise, 1, 2).copy()
        #batch_TDI_noise= xp.reshape(batch_TDI_noise, (self.batch_size, self.dim, self.n_channels))
        return batch_TDI_noise
    
    def noise_whiten_AET(self, noisy_signal_td_AET, dt, channels=["AE","AE","T"]):
        '''This is vectorised for the AET channels.
            GPU-enabled only!'''
        #FFT the windowed TD signal; obtain freq bins
        
        signal_length= len(noisy_signal_td_AET[0])
        window= xp.asarray(tukey(signal_length, alpha=1/8))
        padded_noisy_signal_td_AET= xp.asarray([self.zero_pad(window*noisy_signal_td) for noisy_signal_td in noisy_signal_td_AET])
        
        noisy_signal_fd_AET= xp.fft.rfft(padded_noisy_signal_td_AET)
        
        signal_length= len(padded_noisy_signal_td_AET[0])
        freq = xp.fft.rfftfreq(signal_length, dt)
        freq[0]=freq[1]#To avoid NaN in PSD[0]
        
        
        #Divide FD signal by ASD of noise
        PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn="noisepsd_"+channel, return_type="PSD") for channel in channels])
        
        '''Should this be uncommented?'''
        #Removing NaNs from ASD
        #PSD_AE[0]=PSD_AE[1]
        
        scaling_factor= ((PSD_AET)/(2*dt))**-0.5#len(noisy_signal_td)
        whitened_signal_fd_AET= scaling_factor*noisy_signal_fd_AET

        #IFFTing back into the time domain
        return xp.fft.irfft(whitened_signal_fd_AET, n=len(noisy_signal_td_AET[0]))
    
    def cupy_to_tensor(self, xp_arr):
        '''
        Converts cupy arrays to TF tensors using DL packs for zero-copy data exchange
        '''
        return from_dlpack(xp_arr.toDlpack())
    
    def tensor_to_cupy(self, tf_tensor):
        '''
        Converts TF tensors to cupy arrays using DL packs for zero-copy data exchange
        '''
        return xp.from_dlpack(to_dlpack(tf_tensor))

    def declare_generator_params(self):
        #Declare generator parameters
        print("#################################")
        print("####DATA GENERATOR PARAMETERS####")
        print("#Dataset size: ", self.EMRI_params_set_size)
        print("#Batch size: ", self.batch_size)
        print("#Time in years:", self.T)
        print("#n_channels: ", self.n_channels)
        print("#dt: ",self.dt)
        print("#Length of timeseries:", self.dim)
        print("Noise background: ", self.add_noise)
        print("#################################")

