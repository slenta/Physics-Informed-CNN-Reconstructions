from os import name
from matplotlib.pyplot import plot
import netCDF4 as nc
import numpy as np
import pylab as plt
import h5py

ds = nc.Dataset('/home/simon/Master-Arbeit/Asi_maskiert/original_masks/Maske_2020.nc')
    
#extract the variables from the file
lon = ds['lon'][:]
lat = ds['lat'][:]
time = ds['time'][:]
sst = ds['sao']
sst = np.float(sst[:])


