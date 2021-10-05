
from os import name
import netCDF4 as nc
import numpy as np
import pylab as plt

#name the filepath
fn = 'masken/tos_r8_mask_en4_1970.nc'
name = 'Asmaske_1970'
ds = nc.Dataset(fn)

#extract the variables from the file
lon = ds['lon'][:]
lat = ds['lat'][:]
time = ds['time'][:]
sst = ds['tos'][:]

#complete sst data to create symmetric shape
rest = np.ones((754, 36, 256)) * 9999
sst_new = np.concatenate((sst, rest), axis=1)

#plot ssts in 2d plot
pixel_plot = plt.figure()
pixel_plot = plt.imshow(sst_new[1])
plt.colorbar(pixel_plot)
plt.savefig('masken/' + name + '.pdf')
plt.show()

print(lon[0][0])
print(sst_new.shape)