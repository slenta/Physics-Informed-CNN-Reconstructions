from base64 import decode
import config as cfg
from dataloader import MaskDataset
from utils.netcdfloader import InfiniteSampler
from torch.utils.data import DataLoader
from preprocessing import preprocessing
import h5py
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

cfg.set_train_args()

path_1 = '../Asi_maskiert/original_masks/Maske_2020_newgrid.hdf5'
path_2 = '../Asi_maskiert/original_masks/Maske_1970_newgrid.hdf5'
path_3 = '../Asi_maskiert/original_masks/Maske_1970_newgrid.nc'
path_4 = '../Asi_maskiert/original_image/Image_3d_1958_2020_newgrid.nc'

da = xr.load_dataset(path_4, decode_times=False)
ds = xr.load_dataset(path_3, decode_times=False)
time_var = da.time
da['time'] = netCDF4.num2date(time_var[:],time_var.units)

da_monthly = da.groupby('time.month').mean('time')
sst_mean = da_monthly.thetao.values
sst = ds.tho.values

fc = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
continent_mask = fc.get('tos_sym')


f1 = h5py.File(path_1, 'r')
f2 = h5py.File(path_2, 'r')
f3 = h5py.File(path_2, 'r')

v1 = f1.get('tos_sym')[0, 0, :, :] * continent_mask
v2 = f2.get('tos_sym')[0, 0, :, :] * continent_mask
sst = sst - sst_mean

n = sst.shape
new_im_size = 128

rest = np.zeros((n[0], n[1], new_im_size - n[2], n[3]))
sst = np.concatenate((sst, rest), axis=2)
n = sst.shape
rest2 = np.zeros((n[0], n[1], n[2], new_im_size - n[3]))
sst = np.concatenate((sst, rest2), axis=3)
sst = sst * continent_mask

v3 = sst[0, 0, :, :]

fig = plt.figure(figsize=(12, 4), constrained_layout=True)
plt.subplot(1, 3, 1)
plt.title('Binary Mask: January 2020')
current_cmap = plt.cm.jet
current_cmap.set_bad(color='grey')
im1 = plt.imshow(v1, cmap=current_cmap, vmin=-1, vmax=1, aspect='auto', interpolation=None)
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 2)
plt.title('Binary Mask: January 1970')
im2 = plt.imshow(v2, cmap = 'jet', vmin=-1, vmax=1, aspect = 'auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 3)
plt.title('Observations: January 1970')
im3 = plt.imshow(v3, cmap='jet', vmin=-3, vmax=3, aspect='auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
plt.colorbar(label='Temperature in °C')
fig.savefig('../Asi_maskiert/pdfs/plan.pdf', dpi = fig.dpi)
plt.show()

