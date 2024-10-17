'''
This script is used to plot already generated reconstructions from the
get_reconstructions script.
'''

import numpy as np
import matplotlib.pyplot as plt

from few.utils.constants import YRSID_SI
import matplotlib.pyplot as plt
import os
from scipy import signal

#Load data
val_EMRI_fname= "Val_X_EMRIs_NORMALISED.npy"
pred_EMRI_fname= "Val_pred_EMRIs.npy"
X_EMRIs= np.load(val_EMRI_fname, allow_pickle=True) 
y_pred_EMRIs= np.load(pred_EMRI_fname, allow_pickle=True) 

#Load the normalising arrays
max_abs_tensor= np.array([0.9098072, 0.5969127]).reshape(2,1)
X_EMRIs= X_EMRIs*max_abs_tensor

#Define dataset parameters
dt=10
no_pts= X_EMRIs.shape[2]
T= X_EMRIs.shape[2]*10/YRSID_SI#years

# Bandpass the signal in the time domain before FFTing
'''Low and high frequencies need to be chosen carefully. There are limits on how high the upper limit can be.'''
lowcut = 1e-4  # Low frequency in Hz
highcut = 0.25e-1 # High frequency in Hz
fs = 1/dt  # Sampling frequency in Hz

# Normalize the frequencies to the Nyquist frequency
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist

# Set the Butterworth bandpass filter
sos = signal.butter(2, [low, high], analog=False, btype='band', output='sos')

#Bandpass
y_pred_EMRIs_bandpassed= signal.sosfilt(sos, y_pred_EMRIs)

#Do the FFT
f= np.fft.rfftfreq(n=no_pts, d=dt)
'''Should we be using a Tukey window first?'''
y_pred_EMRIs_bandpassed_FFT=np.abs(np.fft.rfft(y_pred_EMRIs_bandpassed))**2
X_EMRIs_FFT= np.abs(np.fft.rfft(X_EMRIs))**2

#Plot the Y true, Y predicted, and residuals
'''Do something like 2 rows, 3 columns. Row 1 is for the A channel, row 2 the E channel'''
ncols=X_EMRIs.shape[0]
nrows= X_EMRIs.shape[1]

fig, axs= plt.subplots(nrows=nrows, ncols=ncols, sharex=True, gridspec_kw={"hspace":0})
fig.set_size_inches(15, 8)

#Plot inputs, predictions and residuals for each column of the subplot
for col in range(ncols):#subplot, axs.flatten()
  axs[0,col].set(title="EMRI "+ str(col+1))
  axs[-1,col].set_xlabel("Frequency, Hz")
  for channel in range(nrows):
    axs[channel,col].loglog(f, X_EMRIs_FFT[col, channel, :], "b", label="True EMRI", alpha=1)    
    axs[channel,col].loglog(f, y_pred_EMRIs_bandpassed_FFT[col, channel, :], "r", label="Pred. EMRI, bandpassed", alpha=0.6)
    axs[channel,col].set_xlim(1e-4, 1e-1)


#Label the subplots
fig.suptitle('Reconstructions of validation EMRI in TDI AE, SNRs [60,100]')
axs[0,0].set(ylabel="Whitened power spectrum, A chan.")
axs[1,0].set(ylabel="Whitened power spectrum, E chan.")
plt.yscale("log")
plt.xscale("log")
plt.legend()

plt.savefig("testing_data_fd_reconstructions.png")
