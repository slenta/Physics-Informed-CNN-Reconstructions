from os import name
from matplotlib.pyplot import plot
import netCDF4 as nc
import numpy as np
import pylab as plt
import h5py
import xarray as xr
from matplotlib import pyplot as plt

ds = nc.Dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_masks/Maske_2020.nc')
#ds = nc.Dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_image/Assimilation_1958_2020.nc')
#extract the variables from the file
#lon = ds['lon'][:]
#lat = ds['lat'][:]
#time = ds['time'][:]
sst = ds['sao']
#sst = np.float(sst[:])
#image =  xr.load_dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_image/Assimilation_1958_2020.nc', decode_times=False)
#sst = image.tos.values
print(sst[0])
print(sst.shape)
print('')
mask = xr.load_dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_masks/Maske_2020.nc', decode_times=False)
sao = mask.sao.values
sao = sao.astype(float)
print(sao[0])