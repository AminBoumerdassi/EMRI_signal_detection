'''
This script contains custom callbacks such as one that makes the model test on LISA noise at the end of each epoch.
'''
from tensorflow.keras.callbacks import Callback
import cupy as xp
import numpy as np
from numpy.random import default_rng

class TestOnNoise(Callback):
    def __init__(self, model,  generator):
        self.model = model
        self.generator = generator
        self.rng = default_rng(seed=2022)
        self.losses = []
        
    def on_epoch_end(self, epoch, logs={}):
        """The noise here might really be denoised noise. This could mean one of two things:
            1. The actual output of denoised noise from the denoising AE i.e. something where the timeseries
                is a flat line hovering around zero
            2. A EMRI-free signal with no noise background whatsoever i.e.  an array of zeroes."""


        #Initialise an empty array to store noise samples for ONE batch
        #x_test = xp.empty((self.generator.batch_size, self.generator.n_channels, self.generator.dim))
        x_test= self.generator.get_TDI_noise()
        y_true = x_test

        # #Iterate the noise generation and whitening over ONE batch
        # for i in range(self.generator.batch_size):
        #     """This noise generation no longer makes sense as our input doesn't have LISA noise"""
        #     noise_AET= self.generator.noise_td_AET(self.generator.dim, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])#["AE","AE","T"]
        #     x_test[i,:,:]= self.generator.noise_whiten_AET(noise_AET, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])
        # #Reshape the batch of noise samples for input into the model 
        # x_test= xp.reshape(x_test, (self.generator.batch_size, self.generator.dim, self.generator.n_channels))#.get()

        #convert input from xp array to TF tensor
        x_test= self.generator.cupy_to_tensor(x_test)
        #y_true= self.generator.cupy_to_tensor(y_true)

        #Make a prediction with the model, then calculate the corresponding loss
        y_pred= self.model(x_test, training=False)

        
        ''' Tensorflow's in-built MSE function is janky so let's do it by hand.
            The process is:
            1. Find the difference between y_true and y_pred
            2. Square that difference
            3. Take the mean of the squared difference over axis 1
            4. Sum the array of MSEs across the batch
            5. Divide by the batch size'''

        #Convert prediction from TF tensor to xp array
        y_pred= self.generator.tensor_to_cupy(y_pred)

        batch_loss= xp.sum(xp.mean((y_true-y_pred)**2, axis=1)).get()/self.generator.batch_size

        #State and store the losses from these noise samples
        print("Noise loss: ", batch_loss)        
        self.losses.append(batch_loss)