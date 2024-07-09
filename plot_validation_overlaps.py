import numpy as np
import matplotlib.pyplot as plt

#Load data
val_overlaps= np.load("validation_overlaps.npy", allow_pickle=True)

#Plot network overlaps
plt.hist(val_overlaps, bins=25)

plt.title("$O(h_{pred}, h_{input})_{AE}$")
plt.xlabel("Overlap")
plt.ylabel("Frequency")

plt.savefig("validation_overlaps.png")