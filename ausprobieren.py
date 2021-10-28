from os import name
from matplotlib.pyplot import plot
import netCDF4 as nc
import numpy as np
import pylab as plt
import h5py
import xarray as xr
from matplotlib import pyplot as plt
import torch

import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from dataloader import MaskDataset
from util.io import load_ckpt
from util.io import save_ckpt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ds = nc.Dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_masks/Maske_2020.nc')
#ds = nc.Dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_image/Assimilation_1958_2020.nc')
#extract the variables from the file
#lon = ds['lon'][:]
#lat = ds['lat'][:]
#time = ds['time'][:]
sst = ds['sao']
#sst = np.float(sst[:])
image =  xr.load_dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_image/Assimilation_1958_2020.nc', decode_times=False)
sst = image.tos.values
#print(sst[0])
print(sst.shape)
print('')
maske = xr.load_dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_masks/Maske_2020.nc', decode_times=False)
sao = maske.sao.values

f_mask = h5py.File( '/home/simon/Master-Arbeit/Asi_maskiert/original_masks/Maske_2020.hdf5', 'r')
mask_data = f_mask.get('tos_sym')

mask = torch.from_numpy(sao)
print(mask.shape)
mask = mask.unsqueeze(0)
print(mask.shape)
b = mask[0, 0, :,:]
mask = b.repeat(3, 1, 1)
print(mask.shape)

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


model = PartialConv()
