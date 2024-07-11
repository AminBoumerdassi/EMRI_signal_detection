import numpy as np
import matplotlib.pyplot as plt

#Load data
val_overlaps= np.load("validation_overlaps.npy", allow_pickle=True)

val_mismatches= 1-val_overlaps

#Calculate percentile of mismatches and plot as lines
percentiles= (1,5,50,95,99)
xpoints= np.percentile(val_mismatches, percentiles)
percentile_labels= ["Percentile "+str(i) for i in percentiles]
percentile_colours= ["r","g","b","y","black"]

#Plot network overlaps
start=np.percentile(val_mismatches, 0.01)
end=np.percentile(val_mismatches, 99.9)
logbins= np.logspace(np.log10(start),np.log10(end),20)

counts_and_bins=plt.hist(val_mismatches, bins=logbins)

#plt.bar(logbins[:-1], hist, widths)
plt.xscale('log')

plt.title("$M(h_{pred}, h_{input})_{AE}$ of validation EMRIs")
plt.xlabel("Network mismatch")
plt.ylabel("Frequency")


for i in range(len(percentiles)):
    plt.vlines(xpoints[i],counts_and_bins[0].min(),counts_and_bins[0].max(), label=percentile_labels[i],color=percentile_colours[i], linestyle="--")
plt.legend()


plt.savefig("validation_overlaps.png")