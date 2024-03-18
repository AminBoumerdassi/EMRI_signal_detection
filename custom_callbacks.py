from tensorflow.keras.callbacks import Callback
import cupy as xp
import numpy as np
from numpy.random import default_rng


class TestOnNoise(Callback):
    def __init__(self, model,  generator):#, y_test,x_test,
        self.model = model
        #self.x_test = x_test
        #self.y_test = y_test
        self.generator = generator
        self.rng = default_rng(seed=2022)
        self.losses = []
        
    def on_epoch_end(self, epoch, logs={}):
        ''' This is not correct as the noise should be coloured by the LISA PSD.
            This assumes the noise is white. Use the generator's noise generator and whitener.'''
        x_test = xp.empty((self.generator.batch_size, self.generator.n_channels, self.generator.dim))
            
            
            
            
            #x_test[:,]=[self.generator.noise_td_AET(self.generator.dim, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels]) for i in range(self.generator.batch_size)]
            
        for i in range(self.generator.batch_size):
            noise_AET= self.generator.noise_td_AET(self.generator.dim, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])#["AE","AE","T"]
            #noisy_signal_AET= xp.asarray(waveform)+noise_AET

            x_test[i,:,:]= self.generator.noise_whiten_AET(noise_AET, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])#
            #x_test= xp.reshape(x_test, (self.generator.batch_size, self.generator.dim, self.generator.n_channels))
            #y= x_test.get()

            
            #np.empty((self.generator.batch_size, self.generator.n_channels, self.generator.dim))
            
            #self.rng.normal(size=(self.generator.batch_size, self.generator.dim, self.generator.n_channels))
            
#             for i in range(self.generator.batch_size):
#                 #noise_AET= self.rng.normal(size=(self.generator.dim, self.generator.n_channels))

#                 x_test[i,:,:]= self.rng.normal(size=(self.generator.dim, self.generator.n_channels))
#                 x_test= xp.reshape(x_test, (self.generator.batch_size, self.generator.dim, self.generator.n_channels))
#                 #y= x_test.get()
        x_test= xp.reshape(x_test, (self.generator.batch_size, self.generator.dim, self.generator.n_channels)).get()
        y_pred = self.model.evaluate(x_test, x_test)#, self.y_test
        print("Noise loss: ", y_pred)
            #print("Pure noise loss: {}".format(logs["loss"]))
        
        self.losses.append(y_pred)
            #print('y predicted: ', y_pred)

#     def on_epoch_end(self, epoch, logs={}):
#         x_test = xp.empty((self.generator.batch_size, self.generator.n_channels, self.generator.dim))
#         for i in range(self.generator.batch_size):
#             noise_AET= self.generator.noise_td_AET(self.generator.dim, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])#["AE","AE","T"]
#             #noisy_signal_AET= xp.asarray(waveform)+noise_AET

#             x_test[i,:,:]= self.generator.noise_whiten_AET(noise_AET, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])#
#             x_test= xp.reshape(x_test, (self.generator.batch_size, self.generator.dim, self.generator.n_channels))
#             #y= x_test.get()

#             y_pred = self.model.predict(x_test.get())#, self.y_test
#             print('y predicted: ', y_pred)
