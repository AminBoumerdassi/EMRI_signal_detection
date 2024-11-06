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
      self.conv1= nn.Conv1d(2, 32, 65, stride=8, padding=59)#29, 59
      self.conv2= nn.Conv1d(32, 64, 65, stride=8, groups=8, padding=59)#29, 59
      '''Try groups=16 since we actually have twice as many input features at the deeper layers'''
      self.conv3= nn.Conv1d(64, 64, 65, stride=8, groups=8, padding=59)
      self.deconv3= nn.ConvTranspose1d(64, 64, 65, stride=8, groups=8, padding=59, output_padding=5)
      self.deconv2= nn.ConvTranspose1d(64, 32, 65, stride=8, groups=8, padding=59,output_padding=4)#padding=29,output_padding=1 OR padding=59,output_padding=4
      self.deconv1= nn.ConvTranspose1d(32, 2, 65, stride=8, padding=59, output_padding=5)#padding=#29, output_padding=1 OR padding=59, output_padding=5
      
      self.chan_shuffle_16= nn.ChannelShuffle(16)
      self.chan_shuffle_8= nn.ChannelShuffle(8)

      #self.bn_2= nn.BatchNorm1d(32)
      #self.bn_3= nn.BatchNorm1d(64)
      #self.de_bn_2= nn.BatchNorm1d(32)

      self.group_norm_1= nn.GroupNorm(8, 32)
      self.group_norm_2= nn.GroupNorm(8, 64)
      self.degroup_norm_2= nn.GroupNorm(8, 32)

      self.dropout_1= nn.Dropout(p=0.2)
      self.dropout_2= nn.Dropout(p=0.2)
      self.dropout_3= nn.Dropout(p=0.2)
      self.dedropout_3= nn.Dropout(p=0.2)
      self.dedropout_2= nn.Dropout(p=0.2)
      self.dedropout_1= nn.Dropout(p=0.2)


    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      '''Can we retry the batch/group/layer norm just one more time??'''
      enc_1= F.leaky_relu(self.conv1(x))
      enc_1= self.dropout_1(enc_1)
      #x= self.bn_2(x)
      #x= self.group_norm_1(x)
      #x= F.leaky_relu(x)#, negative_slope=0.1

      #x= nn.ZeroPad1d((29,29))(x)
      enc_2= F.leaky_relu(self.conv2(enc_1))
      enc_2= self.dropout_2(enc_2)
      #x= self.bn_3(x)
      #x= self.group_norm_2(x)
      #x= F.leaky_relu(x)#, negative_slope=0.1

      #x= self.chan_shuffle_16(x).detach()

      '''Could these deeper layers benefit from smaller kernel sizes?
      My intuition is that smaller kernels force that layer to pick up local features.'''
      enc_3= F.leaky_relu(self.conv3(enc_2))
      enc_3= self.dropout_3(enc_3)
      #x= F.leaky_relu(x)

      #x= self.chan_shuffle_16(x).detach()

      dec_3= F.leaky_relu(self.deconv3(enc_3))
      dec_3= self.dedropout_3(dec_3)
      # x= F.leaky_relu(x)

      #x= self.chan_shuffle_8(x).detach()
      dec_2= F.leaky_relu(self.deconv2(dec_3+enc_2))
      dec_2= self.dedropout_2(dec_2)
      #x= self.de_bn_2(x)
      #x= self.degroup_norm_2(x)
      #x= F.leaky_relu(x)#, negative_slope=0.1
      #x= nn.ZeroPad1d((29,29))(x)

      dec_1= self.deconv1(dec_2+enc_1)

      '''x= self.conv1(x)
      #x= self.bn_2(x)
      #x= self.group_norm_1(x)
      x= F.leaky_relu(x)#, negative_slope=0.1

      #x= nn.ZeroPad1d((29,29))(x)
      x= self.conv2(x)
      #x= self.bn_3(x)
      #x= self.group_norm_2(x)
      x= F.leaky_relu(x)#, negative_slope=0.1

      #x= self.chan_shuffle_16(x).detach()

      # x= self.conv3(x)
      # x= F.leaky_relu(x)

      #x= self.chan_shuffle_16(x).detach()

      # x= self.deconv3(x)
      # x= F.leaky_relu(x)

      #x= self.chan_shuffle_8(x).detach()
      x= self.deconv2(x)
      #x= self.de_bn_2(x)
      #x= self.degroup_norm_2(x)
      x= F.leaky_relu(x)#, negative_slope=0.1
      #x= nn.ZeroPad1d((29,29))(x)


      #x= nn.ZeroPad1d((29,29))(x)
      x= self.deconv1(x)'''

      return dec_1#x
    
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
