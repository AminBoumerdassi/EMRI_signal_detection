import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvAE(nn.Module):
    def __init__(self):
      '''Defines all the layers of the
         model. This is not sequential!'''
      super(ConvAE, self).__init__()#what does this line mean?
      

      '''PyTorch is a bit janky when it comes to padding in conv and tranpose conv layers.
         Zero-padding will have to be done by hand. Likely there are wrapper functions that
         people have written to do this. Find these!
         
         OR, maybe a different padding mode could be acceptable here?'''
      self.conv1= nn.Conv1d(2, 64, 65, stride=8, dilation=1, padding=29)#, padding_mode='circular'
      self.conv2= nn.Conv1d(64, 64, 65, stride=8, dilation=1, padding=29)#, padding_mode='circular'

      self.conv3= nn.ConvTranspose1d(64, 64, 65, stride=8, dilation=1, padding=29, output_padding=1)#
      self.conv4= nn.ConvTranspose1d(64, 2, 65, stride=8, dilation=1, padding=29, output_padding=1)#

    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      
      x= self.conv1(x)
      x= F.leaky_relu(x)
      x= self.conv2(x)
      x= F.leaky_relu(x)

      x= self.conv3(x)
      x= F.leaky_relu(x)
      x= self.conv4(x)

      return x
    
    def padding_size(stride, L_out, L_in, dilation, kernel_size):
      return np.ceil(0.5*(stride*(L_out-1)-L_in+dilation*(kernel_size-1)+1), dtype=int)


#my_nn = ConvNet()
#print(my_nn)







# def build_model(len_seq=2**22, n_channels=2,hidden_activation= "relu", output_activation= "sigmoid", strides=8):
#     '''Let's experiment with making this model deeper: more layers; and test out leaky relu.
#     Deeper layers may benefit from smaller strides.
#     Even weirder idea: increase the filter size with depth - it won't ruin the bottleneck

#     Why not experiment with normalising inputs between [0,1] rather than [-1,1]?
#     Reasons for:
#     1. can just use relu activations without worrying about negative inputs causing dying relus
#     2. strains are obviously not solely +ve but the normalisation can just be undone after predictions.
#     '''
#     model = Sequential()
#     model.add(Input(shape=(len_seq,n_channels)))
#     model.add(Conv1D(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
#     model.add(LeakyReLU())
#     #model.add(LayerNormalization())
#     #model.add(BatchNormalization())
#     model.add(Conv1D(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
#     model.add(LeakyReLU())
#     #model.add(LayerNormalization())
#     #model.add(BatchNormalization())
#     model.add(Conv1D(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
#     model.add(LeakyReLU())
#     # model.add(LayerNormalization())
#     # #model.add(BatchNormalization())
#     # model.add(Conv1D(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
#     # model.add(LeakyReLU())
#     # #model.add(LayerNormalization())

#     """should the output of the encoder also be +ve/-ve like the input?"""

#     # model.add(Conv1DTranspose(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
#     # model.add(LeakyReLU())
#     # model.add(LayerNormalization())
#     model.add(Conv1DTranspose(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
#     model.add(LeakyReLU())
#     #model.add(LayerNormalization())
#     #model.add(BatchNormalization())
#     model.add(Conv1DTranspose(64,kernel_size=64,activation=hidden_activation, strides=strides, padding='same'))
#     model.add(LeakyReLU())
#     #model.add(LayerNormalization())
#     #model.add(BatchNormalization())
#     model.add(Conv1DTranspose(n_channels,kernel_size=64, strides=strides, padding='same'))#this layer may be redundant
#     '''In some sources, the final layer of a conv AE is NOT a transpose layer'''
#     #model.add(Conv1DTranspose(n_channels,kernel_size=1, strides=1, padding='same'))#this layer may be redundant
#     model.add(Activation(output_activation, dtype="float32"))#using tanh because of min-max scaling - Change if the preprocessing also changes!

#     return model

