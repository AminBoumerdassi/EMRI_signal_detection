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
      #self.batch_norm1= nn.BatchNorm1d(64)#, eps=0.001
      self.conv2= nn.Conv1d(64, 64, 65, stride=8, dilation=1, padding=29)#, padding_mode='circular'
      #self.batch_norm2= nn.BatchNorm1d(128)#, eps=0.001
      #self.conv3= nn.Conv1d(128, 512, 65, stride=8, dilation=1, padding=29)#, padding_mode='circular'

      #self.deconv3= nn.ConvTranspose1d(512, 128, 65, stride=8, dilation=1, padding=29, output_padding=1)#
      #self.batch_norm3= nn.BatchNorm1d(128)#, eps=0.001
      self.deconv2= nn.ConvTranspose1d(64, 64, 65, stride=8, dilation=1, padding=29, output_padding=1)#
      #self.batch_norm4= nn.BatchNorm1d(64)#, eps=0.001
      self.deconv1= nn.ConvTranspose1d(64, 2, 65, stride=8, dilation=1, padding=29, output_padding=1)#

    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      
      x= self.conv1(x)
      x= F.leaky_relu(x)
      #x= self.batch_norm1(x)
      x= self.conv2(x)
      x= F.leaky_relu(x)
      #x= self.batch_norm2(x)
      #x= self.conv3(x)
      #x= F.leaky_relu(x)


      #x= self.deconv3(x)
      #x= F.leaky_relu(x)
      #x= self.batch_norm3(x)
      x= self.deconv2(x)
      x= F.leaky_relu(x)
      #x= self.batch_norm4(x)
      x= self.deconv1(x)

      return x
    
    def padding_size(stride, L_out, L_in, dilation, kernel_size):
      return np.ceil(0.5*(stride*(L_out-1)-L_in+dilation*(kernel_size-1)+1), dtype=int)
    