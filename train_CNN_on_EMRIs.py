'''
This script trains a CNN-autoencoder using the EMRI data generator, records losses, and plots them.
It also uses some custom callbacks for testing on noise at the end of each epoch.
'''

from EMRI_generator_TDI import EMRIGeneratorTDI

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
import numpy as np
from sklearn.model_selection import train_test_split
#from custom_callbacks import TestOnNoise

# GPU check
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Turn off some debugging APIs for faster training
# torch.autograd.profiler.profile(enabled=False)
# torch.autograd.profiler.emit_nvtx(enabled=False)
# torch.autograd.set_detect_anomaly(mode=False)

#Stop PyTorch from being greedy with GPU memory
'''Maybe not needed in torch?'''

#Enable mixed precision
'''Keeping mixed precision off for now as it seems to worsen performance.
   It needs a bit of finesse, and deeper reading into how best to apply
   it for this ML problem.'''
use_amp=False#True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

#Initialise the model and move it to device
model= ConvAE().to(device)

#Setting generator parameters
len_seq= 2**20#23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10#10
T= len_seq*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
n_channels=len(TDI_channels)
add_noise=False#True
seed=2023

#Setting training hyperparameters
batch_size=32
epochs=100
lr=14e-4#8e-4#benchmark at 14e-4
test_size=0.3

#Set some seeds within PyTorch
torch.manual_seed(seed)

#Define loss functions and optimizer
loss_fn= nn.MSELoss().to(device)
optimizer= torch.optim.Adam(params=model.parameters(), lr=lr)

#Initialise the dataset classes for training and val
EMRI_params_dir="training_data/11011_EMRI_params_SNRs_60_100.npy"
EMRI_params= np.load(EMRI_params_dir, allow_pickle=True)
train_params, val_params= train_test_split(EMRI_params, test_size=test_size, random_state=seed)

training_set= EMRIGeneratorTDI(train_params, dim=len_seq, dt=dt, TDI_channels=TDI_channels, add_noise=add_noise, seed=seed)#"training_data/EMRI_params_SNRs_20_100_fixed_redshift.npy"
validation_set= EMRIGeneratorTDI(val_params, dim=len_seq, dt=dt, TDI_channels=TDI_channels, add_noise=add_noise, seed=seed)#"training_data/EMRI_params_SNRs_20_100_fixed_redshift.npy"

#Initialise the data generators as PyTorch dataloaders
training_dataloader= torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)
validation_dataloader= torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, drop_last=True)

#See the architecture of the model
summary(model, input_size=(batch_size, n_channels, len_seq))

#Declare generator's parameters
training_set.declare_generator_params()

#Declare hyperparameters
print("#################################")
print("####TRAINING HYPERPARAMETERS####")
print("#Batch size: ", batch_size)
print("#Learning rate:", lr)
print("#Training proportion of dataset: ", 1-test_size)
print("#No. epochs: ", epochs)
print("#################################")

#Initialise callbacks
#TestOnNoise= TestOnNoise(model, training_and_validation_generator)

#initialise training and validation histories
train_history=[]
val_history=[]

#Load data and the normalising tensor
'''X_EMRIs, y_true_EMRIs = next(iter(validation_dataloader))
max_abs_tensor= torch.tensor([0.9098072, 0.5969127]).reshape(2,1)'''

#Train the model
for t in range(epochs):
    #Initialise variables for measuring time of an epoch
    start= torch.cuda.Event(enable_timing=True)
    end= torch.cuda.Event(enable_timing=True)

    print(f"-----------------------------------\n\t\tEpoch {t+1}/{epochs}\n-----------------------------------")
    start.record()
    train_loop(training_dataloader, model, loss_fn, optimizer, batch_size, train_history, scaler, "cuda", use_amp=use_amp)
    val_loop(validation_dataloader, model, loss_fn, val_history, scaler,  "cuda", use_amp=use_amp)
    end.record()

    #Print time for 1 epoch
    torch.cuda.synchronize()
    print("Epoch time: {:.2f}s\n".format(start.elapsed_time(end)/1000))
    #Save model if lowest loss achieved
    if val_history[-1] == np.array(val_history).min():
        torch.save(model.state_dict(), "model_current.pt")
    
    #Save the current history
    '''np.save("train_history_current.npy",train_history)
    np.save("val_history_current.npy", val_history)'''

    #Quick plot of reconstructions at this particular epoch
    '''y_pred_EMRIs= model(X_EMRIs)
    plt.figure()
    plt.title("Val. data reconstruction at epoch {:}".format(t))
    plt.xlabel("Time, years")
    plt.ylabel("Strain")
    plt.plot(np.linspace(0, T, num=len_seq)[:3000], y_true_EMRIs[0,0,:3000].detach().cpu().numpy(), label="Un-normalised input")
    plt.plot(np.linspace(0, T, num=len_seq)[:3000], y_pred_EMRIs[0,0,:3000].detach().cpu().numpy(), label="Prediction")
    plt.plot(np.linspace(0, T, num=len_seq)[:3000], y_true_EMRIs[0,0,:3000].detach().cpu().numpy()-y_pred_EMRIs[0,0,:3000].detach().cpu().numpy(), label="residual")
    plt.legend(loc='upper right')
    plt.savefig("reconstructions_td_live_epoch_{:}.png".format(t))'''

print("Done!")


#Save the training and val losses
'''EDIT THESE FILE NAMES!'''
np.save("train_history.npy",train_history)
np.save("val_history.npy", val_history)
# np.save("train_history_BS_{:}_lr_0_{:}.npy".format(batch_size, str(lr)[2:]),train_history)
# np.save("val_history_BS_{:}_lr_0_{:}.npy".format(batch_size, str(lr)[2:]), val_history)

#Save model
'''EDIT AS NEEDED!'''
#torch.save(model.state_dict(), "model_BS_{:}_lr_0_{:}.pt".format(batch_size, str(lr)[2:]))
#torch.save(model.state_dict(), "model_current.pt")