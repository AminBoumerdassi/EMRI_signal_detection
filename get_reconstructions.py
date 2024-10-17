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
model_state_dict_dir= "model_current.pt"#"model_group_conv_w_normalisation.pt"

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

#Normalise X
max_abs_tensor= torch.as_tensor([0.9098072, 0.5969127], device="cuda").reshape(2,1)
X_EMRIs= X_EMRIs/max_abs_tensor

#Make predictions with the model
y_pred_EMRIs= model(X_EMRIs)

#Convert everything to numpy arrays
X_EMRIs= X_EMRIs.detach().cpu().numpy()
y_true_EMRIs= y_true_EMRIs.detach().cpu().numpy()
y_pred_EMRIs= y_pred_EMRIs.detach().cpu().numpy()


#Save the example EMRIs and their reconstructions!
np.save("Val_X_EMRIs_NORMALISED.npy",X_EMRIs) 
np.save("Val_pred_EMRIs.npy",y_pred_EMRIs)
