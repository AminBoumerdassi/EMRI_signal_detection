import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal.windows import tukey


class ConvAE(nn.Module):
    def __init__(self):
      '''Defines all the layers of the
         model. This is not sequential!'''
      super(ConvAE, self).__init__()#what does this line mean?
      
      '''Quick idea to test. Why not just double or even 10x the padding? May help with
         getting rid of the artefacts.'''
      self.conv1= nn.Conv1d(2, 32, 65, stride=8, padding=59)#29
      self.conv2= nn.Conv1d(32, 64, 65, stride=8, groups=8, padding=59)#
      '''Try groups=16 since we actually have twice as many input features at the deeper layers'''
      #self.conv3= nn.Conv1d(64, 64, 65, stride=1, groups=16)
      #self.deconv3= nn.ConvTranspose1d(64, 64, 65, stride=1, groups=16, output_padding=0)
      self.deconv2= nn.ConvTranspose1d(64, 32, 65, stride=8, groups=8, padding=59, output_padding=4)#padding=29, output_padding=1
      self.deconv1= nn.ConvTranspose1d(32, 2, 65, stride=8, padding=59, output_padding=5)#padding=29, output_padding=1
      
      self.chan_shuffle_16= nn.ChannelShuffle(16)
      self.chan_shuffle_8= nn.ChannelShuffle(8)

      self.bn_2= nn.BatchNorm1d(32)
      self.de_bn_2= nn.BatchNorm1d(32)
    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      #x= nn.ZeroPad1d((29,29))(x)
      x= self.conv1(x)
      '''For the leaky relu let's try Mica's value of 0.1! Or even tensorflow's default of 0.4'''
      x= F.leaky_relu(x)

      #x= self.bn_2(x)
      #x= nn.ZeroPad1d((29,29))(x)
      x= self.conv2(x)
      x= F.leaky_relu(x)

      #x= self.chan_shuffle_16(x).detach()

      #x= self.conv3(x)
      #x= F.leaky_relu(x)

      #x= self.chan_shuffle_16(x).detach()

      #x= self.deconv3(x)
      #x= F.leaky_relu(x)

      #x= self.chan_shuffle_8(x).detach()
      x= self.deconv2(x)
      x= F.leaky_relu(x)
      #x= nn.ZeroPad1d((29,29))(x)


      #x= self.de_bn_2(x)
      #x= nn.ZeroPad1d((29,29))(x)
      x= self.deconv1(x)
      return x
    
    def padding_size(self, stride, L_out, L_in, dilation, kernel_size):
      '''Something strange about this function. It can sometimes give
         different answers for the same L_in and L_out when expressed
         as ints rather than 2**a and 2**b. Unclear why'''
      return np.ceil(0.5*(stride*(L_out-1)-L_in+dilation*(kernel_size-1)+1))#, dtype=int


class Dilated_ConvAE(nn.Module):
    def __init__(self):
      '''Defines all the layers of the
         model. This is not sequential!'''
      super(ConvAE, self).__init__()#what does this line mean?
      
      '''PyTorch is a bit janky when it comes to padding in conv and tranpose conv layers.
         Zero-padding will have to be done by hand. Likely there are wrapper functions that
         people have written to do this. Find these!
         
         OR, maybe a different padding mode could be acceptable here?'''
      self.conv1= nn.Conv1d(2, 32, 65, stride=8, dilation=1, padding=29)
      self.conv2= nn.Conv1d(32, 64, 65, stride=8, dilation=2, padding=61)
      self.deconv2= nn.ConvTranspose1d(64, 32, 65, stride=8, dilation=1, padding=29, output_padding=1)
      self.deconv1= nn.ConvTranspose1d(32, 2, 65, stride=8, dilation=2, padding=61, output_padding=1)
    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      
      x= self.conv1(x)
      x= F.leaky_relu(x)
      x= self.conv2(x)
      x= F.leaky_relu(x)
      x= self.deconv2(x)
      x= F.leaky_relu(x)
      x= self.deconv1(x)
      return x
    
    def padding_size(self, stride, L_out, L_in, dilation, kernel_size):
      '''Something strange about this function. It can sometimes give
         different answers for the same L_in and L_out when expressed
         as ints rather than 2**a and 2**b. Unclear why'''
      return np.ceil(0.5*(stride*(L_out-1)-L_in+dilation*(kernel_size-1)+1))#, dtype=int
