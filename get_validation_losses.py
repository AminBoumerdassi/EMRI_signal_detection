'''
This tests a trained model on the entire validation set,
and plots a histogram of losses.
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
model_state_dict_dir= "model_BS_32_lr_0_0008_WINDOWED.pt"#model_BS_64_lr_0_0011.pt"

#Load model's weights, architecture, and loss function
model= ConvAE().to(device)
model.load_state_dict(torch.load(model_state_dict_dir))
model.eval()
loss= nn.MSELoss(reduction="none")#reduction="none"

#Specify EMRI generator params
EMRI_params_dir="training_data/11011_EMRI_params_SNRs_60_100.npy"#"training_data/EMRI_params_SNRs_20_100_fixed_redshift.npy"
batch_size=128
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
validation_dataloader= torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

#Iterate predictions over the dataloader, store losses
val_loss_arr_A_E= np.zeros((len(validation_dataloader.dataset), 2))

with torch.no_grad():
    for batch_idx, data in enumerate(validation_dataloader):
        X, y= data
        pred = model(X)
        #Need to do a manual reduction
        val_loss_arr_A_E[batch_idx*batch_size:(batch_idx+1)*batch_size, :] = loss(pred, y).mean(axis=2).detach().cpu().numpy()

#Save losses
"""Currently, this is only actually (1024,2) losses since the dataloader has
    length 1024. Unclear whether this means we have only been training and
     validating on only 1024 EMRIs. """
np.save("validation_losses.npy", val_loss_arr_A_E)


