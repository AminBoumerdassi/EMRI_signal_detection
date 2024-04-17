'''
This script is used to generate and save a dataset of testing data.
Testing data can include things such as:

1. LISA noise
2. Noisy EMRIs not seen by the model
3. Other types of GW sources e.g. MBHBs etc.
4. Glitches
'''

import numpy as np
import cupy as xp
import matplotlib.pyplot as plt

#from tensorflow import keras
from tensorflow.keras import models 
from tensorflow.config import set_logical_device_configuration, list_physical_devices, list_logical_devices
from tensorflow.config.experimental import set_memory_growth

from EMRI_generator_TDI import EMRIGeneratorTDI

#Stop TensorFlow from being greedy with GPU memory
gpus = list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        set_memory_growth(gpu, True)
    logical_gpus = list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


#Specify some variables
model_dir= "model_INSERT_SLURM_ID.keras"

#Specify EMRI generator params
EMRI_params_dir="training_data/EMRI_params_SNRs_20_100_fixed_redshift.npy"
val_dataset_size= 3
batch_size=3#8#This needs to be such that val_dataset_size/batch_size is evenly divisible
dim=2**23#22
TDI_channels="AE"
dt=10
seed=2021
add_noise=False

#Initialise the EMRI generator
validation_data_generator= EMRIGeneratorTDI(EMRI_params_dir=EMRI_params_dir, batch_size=batch_size, dim=dim, dt=dt,  TDI_channels=TDI_channels, seed=seed, add_noise=add_noise)

#Load model
model= models.load_model(model_dir)

#Initialise the arrays containing the reconstruction errors for the various cases of EMRI, LISA noise, etc.
#reconstruction_error_EMRI= np.zeros((val_dataset_size, len(TDI_channels)))#This has separate reconstruction errors for the AET channels so shape (no. EMRIs, no. channels)
#reconstruction_error_noise= np.zeros((val_dataset_size, len(TDI_channels)))

#Generate a batch of whitened validation EMRIs
X_EMRIs, y_true_EMRIs = validation_data_generator.__getitem__(1)#output is actually a TF tensor!

#Generate a batch of whitened LISA noise for testing
X_noise= validation_data_generator.get_TDI_noise().get()

#X_noise = validation_data_generator.noise_whiten_AET(X_noise, dt, channels=["AE","AE"]).get()
y_true_noise =  np.zeros(np.shape(X_EMRIs))#Not whitened but I guess it doesn't need to be

#Make predictions with the model
y_pred_EMRIs= model.predict(X_EMRIs)
y_pred_noise= model.predict(X_noise)

#Calculate the reconstruction error between y true and y pred
'''Reconstructions are not the same as reconstruction errors!'''
#reconstruction_error_EMRI = np.mean(np.square(y_pred_EMRIs - y_true_EMRIs), axis=1)
#reconstruction_error_noise = np.mean(np.square(y_pred_noise - y_true_noise), axis=1)

#Plot the Y true, Y predicted, and residuals (may not be the same as the reconstruction error)
'''Do something like 2 rows, 3 columns. Row 1 is for the A channel, row 2 the E channel'''
fig, axs= plt.subplots(nrows=4, ncols=2)#(ax1, ax2, ax3, ax4, ax5, ax6)

t= np.linspace(0, validation_data_generator.T, num=validation_data_generator.dim)

#Plot the EMRI reconstructions in the A/E channels
axs[0,0].plot(t, X_EMRIs[0,:,0], "purple", label="True noisy EMRI")
axs[1,0].plot(t, y_pred_EMRIs[0,:,0],"g", label="Denoised EMRI")
axs[2,0].plot(t, y_true_EMRIs[0,:,0], "b", label="True noiseless EMRI")
axs[3,0].plot(t, y_true_EMRIs[0,:,0]-y_pred_EMRIs[0,:,0], "r--",label="Residual")# y_pred_EMRIs[0,:,0]-y_true_EMRIs[0,:,0]
         
#axs[1,0].plot(t, y_true_EMRIs[0,:,1], label="True EMRI")
#axs[1,0].plot(t, y_pred_EMRIs[0,:,1], label="Denoised EMRI")
#axs[1,0].plot(t, y_pred_EMRIs[0,:,1]-y_true_EMRIs[0,:,1], label="Residual")
#axs[1,0].legend()

#Plot the LISA noise reconstructions in the A and E channels
axs[0,1].plot(t, X_noise[0,:,0], "purple", label="True LISA noise")
axs[1,1].plot(t, y_pred_noise[0,:,0],"g", label="Denoised noise")
axs[2,1].plot(t, y_true_noise[0,:,0], "b", label="True denoised noise")
axs[3,1].plot(t, y_true_noise[0,:,0]-y_pred_noise[0,:,0], "r", label="Residual")
         
# axs[1,1].plot(t, y_true_noise[0,:,1], label="LISA noise")
# axs[1,1].plot(t, y_pred_noise[0,:,1], label="Denoised noise")
# axs[1,1].plot(t, y_pred_noise[0,:,1]-y_true_noise[0,:,1], label="Residual")
# axs[1,1].legend()

         
#And label the subplots
fig.suptitle('Looking at TDI A')
axs[0,0].set(ylabel="Input", title="Reconstructing EMRIs")
axs[1,0].set(ylabel="Prediction")
axs[2,0].set(ylabel="True output")
axs[3,0].set(ylabel="Residual")
axs[0,1].set(title="Reconstructing noise")
axs[3,0].set(xlabel="Time, years")

plt.savefig("testing_data_reconstructions.png")






# for i in range(int(val_dataset_size/batch_size)):
#     #Generate various types of data e.g. FEW EMRIs, Gaussian noise, all whitened!
#     validation_EMRIs= validation_data_generator.__getitem__(1)[0]
    
#     #Gaussian noise generation
#     validation_noise= xp.zeros((batch_size, len(TDI_channels), dim))
#     for element in range(batch_size):
#         validation_noise[element,:,:]= validation_data_generator.noise_td_AET(dim, dt=dt, channels=noise_channels)
#         validation_noise[element,:,:]= validation_data_generator.noise_whiten_AET(validation_noise[element,:,:], dt=dt, channels=noise_channels)#this only produces 1 sample!
#     validation_noise= np.reshape(validation_noise.get(), (batch_size, dim, len(TDI_channels)))
        
#     #Make predictions with model
#     prediction_EMRIs= model.predict(validation_EMRIs)
#     prediction_noise= model.predict(validation_noise)
#     #Calculate reconstruction errors for different waveforms
#     reconstruction_error_EMRI[i*batch_size:((i+1)*batch_size),:]= np.mean(np.square(validation_EMRIs - prediction_EMRIs), axis=1)#This has separate reconstruction errors for the AET channels so shape (no. EMRIs, no. channels)
#     reconstruction_error_noise[i*batch_size:((i+1)*batch_size),:]= np.mean(np.square(validation_noise - prediction_noise), axis=1)




