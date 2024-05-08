from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Activation, LeakyReLU, BatchNormalization, LayerNormalization
#from tensorflow.keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Cropping1D, ZeroPadding1D,

def build_model(len_seq=2**22, n_channels=2,hidden_activation= "relu", output_activation= "sigmoid", strides=8):
    '''Let's experiment with making this model deeper: more layers; and test out leaky relu.
    Deeper layers may benefit from smaller strides.
    Even weirder idea: increase the filter size with depth - it won't ruin the bottleneck

    Why not experiment with normalising inputs between [0,1] rather than [-1,1]?
    Reasons for:
    1. can just use relu activations without worrying about negative inputs causing dying relus
    2. strains are obviously not solely +ve but the normalisation can just be undone after predictions.
    '''
    model = Sequential()
    model.add(Input(shape=(len_seq,n_channels)))
    model.add(Conv1D(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
    model.add(LeakyReLU())
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    model.add(Conv1D(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
    model.add(LeakyReLU())
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    model.add(Conv1D(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
    model.add(LeakyReLU())
    # model.add(LayerNormalization())
    # #model.add(BatchNormalization())
    # model.add(Conv1D(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
    # model.add(LeakyReLU())
    # #model.add(LayerNormalization())

    """should the output of the encoder also be +ve/-ve like the input?"""

    # model.add(Conv1DTranspose(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
    # model.add(LeakyReLU())
    # model.add(LayerNormalization())
    model.add(Conv1DTranspose(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
    model.add(LeakyReLU())
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    model.add(Conv1DTranspose(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
    model.add(LeakyReLU())
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    model.add(Conv1DTranspose(n_channels,kernel_size=64, strides=strides, padding='same'))#this layer may be redundant
    '''In some sources, the final layer of a conv AE is NOT a transpose layer'''
    #model.add(Conv1DTranspose(n_channels,kernel_size=1, strides=1, padding='same'))#this layer may be redundant
    model.add(Activation(output_activation, dtype="float32"))#using tanh because of min-max scaling - Change if the preprocessing also changes!

    return model





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

#Last layers should be a linear activation function otherwise output will not be correct in scale

# model.add(LSTM(16, activation='tanh', input_shape=(len_seq,n_channels), return_sequences=True))
# model.add(LSTM(8, activation='tanh', return_sequences=False))
# model.add(RepeatVector(len_seq))
# model.add(LSTM(8, activation='tanh', return_sequences=True))
# model.add(LSTM(16, activation='tanh', return_sequences=True))
# model.add(TimeDistributed(Dense(n_channels)))

#Choose an optimiser
#optimizer= Adafactor()
