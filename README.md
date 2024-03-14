# EMRI_signal_detection

This repo uses convolutional autoencoders to perform a signal detection of EMRIs detected by the LISA mission. By framing the signal detection problem in terms of anomaly detection, the model learns to represent time-domain EMRIs in a low-dimensional latent space, and reproduce the original EMRI from this. Ideally, signals other than EMRIs will be poorly reproduced as the trained encoding and decoding process will only work for EMRIs.

Packages/citations: (to-do)

Numpy, cupy, TensorFlow, Scipy, FEW, LisaonGPU, astropy, lisatools, 
