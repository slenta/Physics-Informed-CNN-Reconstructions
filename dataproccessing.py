
import netCDF4 as nc
import numpy as np
import pylab as plt

fn = 'masken/tos_r8_mask_en4_1970.nc'
ds = nc.Dataset(fn)

lon = ds['lon'][:]
lat = ds['lat'][:]
time = ds['time'][:]



print(lon[0:20], lat[0:20], time[0:20])
