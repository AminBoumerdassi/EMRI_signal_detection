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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from test_and_train_loop import *
from model_architecture import ConvAE
from few.utils.constants import YRSID_SI
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


from EMRI_generator_TDI import EMRIGeneratorTDI

# GPU check
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Specify some variables
model_state_dict_dir= "model_BS_64_lr_0_0011.pt"#"model_INSERT_SLURM_ID.pt"

#Load model's weights and architecture
model= ConvAE().to(device)
model.load_state_dict(torch.load(model_state_dict_dir))
model.eval()

#Specify EMRI generator params
EMRI_params_dir="training_data/11011_EMRI_params_SNRs_60_100.npy"#"training_data/EMRI_params_SNRs_20_100_fixed_redshift.npy"
batch_size=4#This needs to be such that val_dataset_size/batch_size is evenly divisible
dim=2**20
TDI_channels="AE"
dt=10
seed=2023
add_noise=False

#Set some seeds
torch.manual_seed(seed)

#Initialise the dataset classes for training and val
EMRI_params_dir="training_data/11011_EMRI_params_SNRs_60_100.npy"
EMRI_params= np.load(EMRI_params_dir, allow_pickle=True)
_, val_params= train_test_split(EMRI_params, test_size=0.3, random_state=seed)

validation_set= EMRIGeneratorTDI(val_params, dim=dim, dt=dt, TDI_channels=TDI_channels, add_noise=add_noise, seed=seed)#"training_data/EMRI_params_SNRs_20_100_fixed_redshift.npy"

#Initialise the data generators as PyTorch dataloaders
validation_dataloader= torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

#Generate one batch of data
X_EMRIs, y_true_EMRIs = next(iter(validation_dataloader))

#Make predictions with the model
y_pred_EMRIs= model(X_EMRIs)

#Convert everything to numpy arrays
X_EMRIs= X_EMRIs.detach().cpu().numpy()
y_true_EMRIs= y_true_EMRIs.detach().cpu().numpy()
y_pred_EMRIs= y_pred_EMRIs.detach().cpu().numpy()

#Plot the Y true, Y predicted, and residuals
'''Do something like 2 rows, 3 columns. Row 1 is for the A channel, row 2 the E channel'''
ncols=batch_size
fig, axs= plt.subplots(nrows=len(TDI_channels), ncols=ncols, sharex=True)#(ax1, ax2, ax3, ax4, ax5, ax6)

t= np.linspace(0, validation_set.T, num=validation_set.dim)

#Plot inputs, predictions and residuals for each column of the subplot
for col in range(ncols):#subplot, axs.flatten()
  axs[0,col].set(title="EMRI "+ str(col+1))
  #If doing reconstructions across the A and E channels:
  for channel in range(len(TDI_channels)):
    axs[channel,col].plot(t, X_EMRIs[col,channel,:], "g", label="True EMRI")
    axs[channel,col].plot(t, y_pred_EMRIs[col,channel,:], "b", label="Pred. EMRI")
    axs[channel,col].plot(t, X_EMRIs[col,channel,:]-y_pred_EMRIs[col,channel,:], "r", label="Residual")

  # axs[0,col].plot(t, X_EMRIs[col,0,:], "purple", label="True EMRI")
  # axs[1,col].plot(t, y_pred_EMRIs[col,0,:], "b", label="Pred. EMRI")
  # axs[2,col].plot(t, X_EMRIs[col,0,:]-y_pred_EMRIs[col,0,:], "r", label="Residual")

# #And label the subplots
fig.suptitle('Reconstructions of validation EMRI in TDI AE, SNRs [60,100]')
axs[0,0].set(ylabel="Whitened strain, A chan.")
axs[1,0].set(ylabel="Whitened strain, E chan.")
axs[-1,0].set(xlabel="Time, years")#, ylabel="Residual"
plt.legend()

plt.savefig("testing_data_reconstructions.png")

#Save the example EMRIs and their reconstructions!
np.save("Val_X_EMRIs.npy",X_EMRIs) 
np.save("Val_pred_EMRIs.npy",y_pred_EMRIs) 

