'''
This script is used to plot already generated reconstructions from the
get_reconstructions script.
'''

import numpy as np
import matplotlib.pyplot as plt

from few.utils.constants import YRSID_SI
import matplotlib.pyplot as plt
import os

#Load data
X_EMRIs= np.load("Val_X_EMRIs.npy", allow_pickle=True) 
y_pred_EMRIs= np.load("Val_pred_EMRIs.npy", allow_pickle=True) 

#Define dataset parameters
dt=10
no_pts= X_EMRIs.shape[2]
T= X_EMRIs.shape[2]*10/YRSID_SI#years

f= np.fft.rfftfreq(n=no_pts, d=dt)

#Plot the Y true, Y predicted, and residuals
'''Do something like 2 rows, 3 columns. Row 1 is for the A channel, row 2 the E channel'''
ncols=X_EMRIs.shape[0]
nrows= X_EMRIs.shape[1]

fig, axs= plt.subplots(nrows=nrows, ncols=ncols, sharex=True,gridspec_kw={"hspace":0})#(ax1, ax2, ax3, ax4, ax5, ax6)

#t= np.linspace(0, T, num=no_pts)

#Plot inputs, predictions and residuals for each column of the subplot
for col in range(ncols):#subplot, axs.flatten()
  axs[0,col].set(title="EMRI "+ str(col+1))
  #If doing reconstructions across the A and E channels:
  for channel in range(nrows):
    axs[channel,col].loglog(f, np.abs(np.fft.rfft(X_EMRIs[col,channel,:]))**2, "b", label="True EMRI", alpha=1)
    
    #Should really window the predictions before FFTing
    axs[channel,col].loglog(f, np.abs(np.fft.rfft(y_pred_EMRIs[col,channel,:]))**2, "r", label="Pred. EMRI", alpha=0.6)
    #axs[channel,col].plot(f, np.fft.rfft(X_EMRIs[col,channel,:])-np.fft.rfft(y_pred_EMRIs[col,channel,:]), "g", label="Residual")

    #axs[channel,col].set(xlabel="Time, years")
    #axs[channel,col].label_outer()

# #And label the subplots
fig.suptitle('Reconstructions of validation EMRI in TDI AE, SNRs [60,100]')
axs[0,0].set(ylabel="Whitened PSD, A chan.")
axs[1,0].set(ylabel="Whitened PSD, E chan.")
axs[-1,0].set(xlabel="Frequency, Hz")
plt.yscale("log")
plt.xscale("log")
plt.legend()

plt.savefig("testing_data_fd_reconstructions.png")
