'''
This script trains a CNN-autoencoder using the EMRI data generator, records losses, and plots them.
It also uses some custom callbacks for testing on noise at the end of each epoch.
'''

from EMRI_generator_TDI import EMRIGeneratorTDI
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Input, Conv1D, Conv1DTranspose, Cropping1D, ZeroPadding1D, Activation
#from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.config import set_logical_device_configuration, list_physical_devices, list_logical_devices
from tensorflow.config.experimental import set_memory_growth
from tensorflow.keras.optimizers import Adafactor, Adam
from few.utils.constants import YRSID_SI
import matplotlib.pyplot as plt
import os
import numpy as np
from custom_callbacks import TestOnNoise

#Stop TensorFlow from being greedy with GPU memory
gpus = list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        set_memory_growth(gpu, True)
    logical_gpus = list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#Enable mixed precision for less GPU-memory-intensive training and increased batch size
mixed_precision.set_global_policy('mixed_float16')# "mixed_bfloat16"


#Setting generator parameters
len_seq= 2**23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10#10
T= len_seq*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
batch_size=8#The code can now handle larger batch sizes like 32 and maybe more
n_channels=len(TDI_channels)
add_noise=False#True

#Setting training hyperparameters
epochs=100#40#150#5#0#600#5


#Model
'''
Ideas for model design:

1. Try very large strides e.g. 256. Will significantly reduce computation.
Could try very large strides on the 1st layer, then smaller strides for the rest 
Question is: will it lead to certain frequencies not being picked up (questionable as convolutions look for spatial not time dependencies)

2.
'''
# model = Sequential()
# model.add(Input(shape=(len_seq,n_channels)))
# model.add(Conv1D(32,kernel_size=128,activation='relu', strides=4, padding='same'))
# model.add(Conv1D(16,kernel_size=128,activation='relu', strides=4, padding='same'))
# model.add(Conv1D(8,kernel_size=128,activation='relu', strides=4, padding='same'))

# model.add(Conv1DTranspose(8,kernel_size=128,activation='relu', strides=4, padding='same'))
# model.add(Conv1DTranspose(16,kernel_size=128,activation='relu', strides=4, padding='same'))
# model.add(Conv1DTranspose(32,kernel_size=128,activation='relu', strides=4, padding='same'))
# model.add(Conv1DTranspose(n_channels,kernel_size=1, strides=1, padding='same'))#this layer may be redundant
# model.add(Conv1DTranspose(n_channels,kernel_size=1, strides=1, padding='same'))#Perhaps this may scale the output?
# model.add(Activation("linear", dtype="float32"))

# '''A high-stride model'''
'''Let's experiment with making this model deeper: more layers; and test out leaky relu.
Deeper layers may benefit from smaller strides.
Even weirder idea: increase the filter size with depth - it won't ruin the bottleneck'''
model = Sequential()
model.add(Input(shape=(len_seq,n_channels)))
model.add(Conv1D(64,kernel_size=64,activation='relu', strides=8, padding='same'))
model.add(Conv1D(64,kernel_size=64,activation='relu', strides=8, padding='same'))
model.add(Conv1D(64,kernel_size=64,activation='relu', strides=8, padding='same'))
#model.add(Conv1D(32,kernel_size=64,activation='relu', strides=8, padding='same'))

#model.add(Conv1DTranspose(32,kernel_size=64,activation='relu', strides=4, padding='same'))
#model.add(Conv1DTranspose(32,kernel_size=64,activation='relu', strides=8, padding='same'))
model.add(Conv1DTranspose(64,kernel_size=64,activation='relu', strides=8, padding='same'))
model.add(Conv1DTranspose(64,kernel_size=64,activation='relu', strides=8, padding='same'))
model.add(Conv1DTranspose(n_channels,kernel_size=64, strides=8, padding='same'))#this layer may be redundant
'''In some sources, the final layer of a conv AE is NOT a transpose layer'''
#model.add(Conv1D(n_channels,kernel_size=1, strides=1, padding='same'))#this layer may be redundant
model.add(Activation("linear", dtype="float32"))


#Last layers should be a linear activation function otherwise output will not be correct in scale

# model.add(LSTM(16, activation='tanh', input_shape=(len_seq,n_channels), return_sequences=True))
# model.add(LSTM(8, activation='tanh', return_sequences=False))
# model.add(RepeatVector(len_seq))
# model.add(LSTM(8, activation='tanh', return_sequences=True))
# model.add(LSTM(16, activation='tanh', return_sequences=True))
# model.add(TimeDistributed(Dense(n_channels)))

#Choose an optimiser
#optimizer= Adafactor()

model.compile(optimizer=Adam(learning_rate=0.00001), loss="mse")#optimizer="Adam", jit_compile=True
model.summary()

#Initialise data generator, and declare its parameters
training_and_validation_generator= EMRIGeneratorTDI(EMRI_params_dir="training_data/11011_EMRI_params_SNRs_60_100.npy", batch_size=batch_size,  dim=len_seq, dt=dt, TDI_channels=TDI_channels, add_noise=add_noise)#"training_data/EMRI_params_SNRs_20_100_fixed_redshift.npy"
training_and_validation_generator.declare_generator_params()

#Initialise callbacks
TestOnNoise= TestOnNoise(model, training_and_validation_generator)
#ModelCheckpoint= ModelCheckpoint(".", save_weights_only=True, monitor='val_loss',
#                                 mode='min', save_best_only=True)

#Train
history = model.fit(training_and_validation_generator, epochs=epochs, validation_data= training_and_validation_generator,  verbose=2)#callbacks=[TestOnNoise],

#Save model
model.save("model_INSERT_SLURM_ID.keras")

#Plot losses
plt.plot(history.epoch, history.history["loss"], "blue", label='Training loss')
plt.plot(history.epoch, history.history["val_loss"], "orange", label='Validation loss')
#plt.plot(history.epoch, TestOnNoise.losses, "green", label="Noise loss")
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("training_and_val_loss.png")

