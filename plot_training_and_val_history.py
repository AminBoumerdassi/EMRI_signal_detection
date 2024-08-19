import numpy as np
import matplotlib.pyplot as plt

train_fname= "train_history_BS_32_lr_0_0008_WINDOWED.npy"
val_fname=  "val_history_BS_32_lr_0_0008_WINDOWED.npy"

train_history= np.load(train_fname, allow_pickle=True)
val_history= np.load(val_fname,  allow_pickle=True)
epochs= np.arange(1, len(val_history)+1, step=1)

plt.plot(epochs, train_history, "blue", label='Training loss')
plt.plot(epochs, val_history, "orange", label='Validation loss')
#plt.plot(history.epoch, TestOnNoise.losses, "green", label="Noise loss")
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("training_and_val_loss_WINDOWED.png")

