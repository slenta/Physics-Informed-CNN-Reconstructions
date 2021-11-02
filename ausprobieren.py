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
#print(sst.shape)
#print('')
#maske = xr.load_dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_masks/Maske_2020.nc', decode_times=False)
#sao = maske.sao.values

#f_mask = h5py.File( '/home/simon/Master-Arbeit/Asi_maskiert/original_masks/Maske_2020.hdf5', 'r')
#mask_data = f_mask.get('tos_sym')

#mask = torch.from_numpy(sao)
#print(mask.shape)
#mask = mask.unsqueeze(0)
#print(mask.shape)
#b = mask[0, 0, :,:]
#mask = b.repeat(3, 1, 1)
#print(mask.shape)

#mask = torch.stack(mask)
#print(mask.shape)

a = np.ones((1, 2, 3))
print(a)
print(np.shape(a))
a = np.repeat(a, 3, axis=0)
print(np.shape(a))
print(a)