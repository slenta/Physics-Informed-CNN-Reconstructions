
from os import name
from matplotlib.pyplot import plot
import netCDF4 as nc
import numpy as np
import pylab as plt
import h5py

#name the filepath
fn_1 = 'Asi_maskiert/original_masks/tos_r8_mask_en4_1970.nc'
name_1 = 'Asmaske_1970'

fn_2 = 'Asi_maskiert/original_masks/tos_r8_mask_en4_2004.nc'
name_2 = 'Asmaske_2004'

fn_3 = 'Asi_maskiert/original_masks/tos_r8_mask_en4_2020.nc'
name_3 = 'Asmaske_2020'

ds_1 = nc.Dataset(fn_1)

def plot_picture(fn, name):
    
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
    pixel_plot = plt.imshow(sst_new[1], vmin=-30, vmax=45)
    plt.colorbar(pixel_plot)
    plt.savefig('Asi_maskiert/images/' + name + '.pdf')

    #create new h5 file with symmetric ssts
    f = h5py.File(name + '.hdf5', 'w')
    tos = f.create_dataset('tos_sym', (754, 256, 256))
    tos = sst_new
    f.close()

    print(sst_new.shape)

plot_picture(fn_1, name_1)
plot_picture(fn_2, name_2)
plot_picture(fn_3, name_3)
    