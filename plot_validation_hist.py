import numpy as np
import matplotlib.pyplot as plt

#Load data
val_loss_arr_A_E= np.load("validation_losses.npy", allow_pickle=True)

#Subplot with each row for each channel
TDI_channels="AE"
ncols=1
fig, axs= plt.subplots(nrows=len(TDI_channels), ncols=ncols, sharex=True)#(ax1, ax2, ax3, ax4, ax5, ax6)

for channel in range(len(TDI_channels)):
    axs[channel].hist(val_loss_arr_A_E[:,channel], bins=50, range=(0,1e-5))
    axs[channel].axvline(np.percentile(val_loss_arr_A_E[:,channel], 5), label="5th percentile",color="green", linestyle="--")
    axs[channel].axvline(np.percentile(val_loss_arr_A_E[:,channel], 95), label="95th percentile",color="red", linestyle="--")


fig.suptitle('Histogram of EMRI validation losses, TDI AE')
axs[0].set(ylabel="Frequency, A chan.")
axs[1].set(ylabel="Frequency, E chan", xlabel="Loss")
plt.legend()
plt.savefig("val_loss_hist_A_E.png")

# #plot histograms for the A and E channels
# plt.figure()
# plt.hist(val_loss_arr_A_E[:,0], bins=50, range=(0,1e-5))
# plt.xlabel("Mean squared error")
# plt.ylabel("Frequency")
# plt.title("Histogram of validation losses in A channel")

# plt.axvline(np.percentile(val_loss_arr_A_E[:,0], 5), label="5th percentile",color="green", linestyle="--")
# plt.axvline(np.percentile(val_loss_arr_A_E[:,0], 95), label="95th percentile",color="red", linestyle="--")
# plt.legend()

# plt.savefig("val_loss_hist_A_channel.png")


# plt.figure()
# plt.hist(val_loss_arr_A_E[:,1], bins=50, range=(0,1e-5))
# plt.xlabel("Mean squared error")
# plt.ylabel("Frequency")
# plt.title("Histogram of validation losses in E channel")

# plt.axvline(np.percentile(val_loss_arr_A_E[:,1], 5), label="5th percentile",color="green", linestyle="--")
# plt.axvline(np.percentile(val_loss_arr_A_E[:,1], 95), label="95th percentile",color="red", linestyle="--")
# plt.legend()

# plt.savefig("val_loss_hist_E_channel.png")