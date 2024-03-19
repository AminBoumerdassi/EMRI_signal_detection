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
        #Initialise an empty array to store noise samples for ONE batch
        x_test = xp.empty((self.generator.batch_size, self.generator.n_channels, self.generator.dim))
        #Iterate the noise generation and whitening over ONE batch
        for i in range(self.generator.batch_size):
            noise_AET= self.generator.noise_td_AET(self.generator.dim, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])#["AE","AE","T"]
            x_test[i,:,:]= self.generator.noise_whiten_AET(noise_AET, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])
        #Reshape the batch of noise samples for input into the model 
        x_test= xp.reshape(x_test, (self.generator.batch_size, self.generator.dim, self.generator.n_channels)).get()
        y_pred = self.model.evaluate(x_test, x_test)
        #State and store the losses from these noise samples
        print("Noise loss: ", y_pred)        
        self.losses.append(y_pred)