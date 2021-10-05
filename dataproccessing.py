
import netCDF4 as nc
import numpy as np
import pylab as plt

fn = 'masken/tos_r8_mask_en4_1970.nc'
ds = nc.Dataset(fn)

print(ds)