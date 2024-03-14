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
#T is calculated through the choice of dt and the desired length of the timeseries
len_seq= 2**22
dt=10#10
T= len_seq*dt/round(YRSID_SI)
TDI_channels="AE"

#Notes on resource-appropriate batch size (it depends on the model size)
#Max no. on A1001g.5gb: 5 EMRIs, 8 month long
#Max no. of A100:1 : 5 EMRIs, 8 month long
#Max no. of A100:2 : 



batch_size=8#5#on A1001g.5gb, max of 3 1y EMRIs. On A100(40GB), current max of 10(?) 1y EMRIs
n_channels=len(TDI_channels)#channel 1 is real strain component, channel 2 the imaginary component

#Setting training hyperparameters
epochs=150#5#0#600#5


#An empty model
model = Sequential()
model.add(Input(shape=(len_seq,n_channels)))
model.add(Conv1D(8,kernel_size=128,activation='tanh', strides=4, padding='same'))
model.add(Conv1D(8,kernel_size=128,activation='tanh', strides=4, padding='same'))
#model.add(Conv1D(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
#model.add(Conv1D(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
#model.add(Conv1D(8,kernel_size=32,activation='tanh', strides=2, padding='same'))

#model.add(Conv1DTranspose(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
#model.add(Conv1DTranspose(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
#model.add(Conv1DTranspose(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
model.add(Conv1DTranspose(8,kernel_size=128,activation='tanh', strides=4, padding='same'))
model.add(Conv1DTranspose(8,kernel_size=128,activation='tanh', strides=4, padding='same'))
model.add(Conv1DTranspose(n_channels,kernel_size=1, strides=1, padding='same'))#this layer may be redundant
model.add(Activation("linear", dtype="float32"))

# model.add(Conv1D(32,kernel_size=32,activation='tanh', strides=1, padding='same'))
# model.add(Conv1D(16,kernel_size=32,activation='tanh', strides=2, padding='same'))
# model.add(Conv1D(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
# model.add(Conv1D(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
# model.add(Conv1D(8,kernel_size=32,activation='tanh', strides=2, padding='same'))

# model.add(Conv1DTranspose(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
# model.add(Conv1DTranspose(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
# model.add(Conv1DTranspose(8,kernel_size=32,activation='tanh', strides=2, padding='same'))
# model.add(Conv1DTranspose(16,kernel_size=32,activation='tanh', strides=2, padding='same'))
# model.add(Conv1DTranspose(32,kernel_size=32,activation='tanh', strides=1, padding='same'))
# model.add(Conv1DTranspose(n_channels,kernel_size=1, strides=1, padding='same'))
# model.add(Activation("linear", dtype="float32"))


#Last layers should be a linear activation function otherwise output will not be correct in scale

# model.add(LSTM(16, activation='tanh', input_shape=(len_seq,n_channels), return_sequences=True))
# model.add(LSTM(8, activation='tanh', return_sequences=False))
# model.add(RepeatVector(len_seq))
# model.add(LSTM(8, activation='tanh', return_sequences=True))
# model.add(LSTM(16, activation='tanh', return_sequences=True))
# model.add(TimeDistributed(Dense(n_channels)))

#Choose an optimiser
#optimizer= Adafactor()

model.compile(optimizer=Adam(), loss="mse")#optimizer="Adam", jit_compile=True
model.summary()

#Initialise data generator, and declare its parameters
training_and_validation_generator= EMRIGeneratorTDI(batch_size=batch_size,  dim=len_seq, dt=dt, TDI_channels=TDI_channels)#n_channels=n_channels,
training_and_validation_generator.declare_generator_params()

#Initialise callbacks
TestOnNoise= TestOnNoise(model, training_and_validation_generator)
#ModelCheckpoint= ModelCheckpoint(".", save_weights_only=True, monitor='val_loss',
#                                 mode='min', save_best_only=True)

#Train
history = model.fit(training_and_validation_generator, epochs=epochs, validation_data= training_and_validation_generator, verbose=2, callbacks=[TestOnNoise])#,use_multiprocessing=True, workers=6)

#Save model
model.save("model_INSERT_SLURM_ID.keras")

#Plot losses
plt.plot(history.epoch, history.history["loss"], "blue", label='Training loss')
plt.plot(history.epoch, history.history["val_loss"], "orange", label='Validation loss')
plt.plot(history.epoch, TestOnNoise.losses, "green", label="Noise loss")
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("training_and_val_loss.png")

