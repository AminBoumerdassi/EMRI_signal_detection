# EMRI_signal_detection

This project uses ML techniques - namely, the convolutional autoencoder - to perform a signal detection of [extreme mass-ratio inspirals](https://en.wikipedia.org/wiki/Extreme_mass_ratio_inspiral) which are expected to be detected by the [LISA mission](https://en.wikipedia.org/wiki/Laser_Interferometer_Space_Antenna). In this context, conventional signal detection techniques such as matched filtering are impractical owing to the large number of parameters required to model EMRIs.

The autoencoder learns how to represent EMRI signals in a low-dimensional latent space through non-linear transformations on the input data which is the GW strain in the time domain. Ideally, these non-linear transformations will lead to accurate reconstructions of EMRIs, and poor reconstructions of other types of signals. Hence, the EMRI signal detection problem is framed as one of anomaly detection.


## Packages/citations: (May be incomplete)

- Numpy
- Cupy
- TensorFlow
- Scipy
- [FastEMRIWaveforms](https://bhptoolkit.org/FastEMRIWaveforms/html/index.html)
- [fastlisaresponse](https://github.com/mikekatz04/lisa-on-gpu)
- AstroPy
- [LISAanalysistools](https://github.com/mikekatz04/LISAanalysistools)
