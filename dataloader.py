import math
from typing import Type
from matplotlib.pyplot import plot
import pylab as plt
import numpy as np
import netCDF4 as nc
import h5py
import torch
from torch._C import dtype
#from torch._C import float32
#from torch._C import float32
#from torch._C import double
from torch.utils import data
from torch.utils.data import Dataset
import xarray as xr



#dataloader and dataloader

def preprocessing(path, name, year, type, plot):
    
    #ds = nc.Dataset(path + 'Asi_maskiert/masked_images/'name + '.nc')
    ds = xr.load_dataset(path + name + year + '.nc', decode_times=False)

    #extract the variables from the file
    if type == 'mask':
        sst = ds.tho.values[:, 0, :, :]
        x = np.isnan(sst)
        #sst[x] = -9999
        for i in range(12):
            for j in range(220):
                for k in range(256):
                    if np.isnan(sst[i, j, k]) == True:
                        sst[i, j, k] = 0
                    else:
                        sst[i, j, k] = 1
        
        #print(np.shape(sst))
        rest = np.ones((12, 36, 256)) * 0
        sst_new = np.concatenate((sst, rest), axis=1)
        sst_new = np.repeat(sst_new, 63, axis=0)
        #print(np.shape(sst_new))
        #create new h5 file with symmetric ssts
        f = h5py.File(path + name + year + '.hdf5', 'w')
        dset1 = f.create_dataset('tos_sym', (756, 256, 256), dtype = 'float32', data = sst_new)
        f.close()

    if type == 'image':
        sst = ds.tos.values
        x = np.isnan(sst)
        n = sst.shape
        sst[x] = 999999
        print(sst.shape)
        #print(np.any(np.isnan(sst)))
        rest = np.ones((n[0], n[2] - n[1], n[2])) * 999999
        sst_new = np.concatenate((sst, rest), axis=1)
         
        #create new h5 file with symmetric ssts
        f = h5py.File(path + name + year + '.hdf5', 'w')
        dset1 = f.create_dataset('tos_sym', (n[0], n[2], n[2]), dtype = 'float32',data = sst_new)
        f.close()

    #plot ssts in 2d plot
    if plot == True:
        pixel_plot = plt.figure()
        pixel_plot = plt.imshow(sst_new[1], vmin = -5, vmax = 5)
        plt.colorbar(pixel_plot)
        plt.savefig('../Asi_maskiert/pdfs/' + name + '.pdf')
        plt.show()
        


#preprocessing('../Asi_maskiert/masked_images/', 'tos_r8_mask_en4_2004', type='image', plot=True)
#preprocessing('../Asi_maskiert/original_masks/', 'Maske_', '2020', type='mask', plot = False)
#preprocessing('../Asi_maskiert/original_image/', 'Image_', '2020', type='image', plot=False)
#preprocessing('../Asi_maskiert/Chris_Daten/', 'Chris_image', type='image', plot=True)
#preprocessing('../Asi_maskiert/Chris_Daten/', 'Chris_masks', type='mask', plot=True)


class MaskDataset(Dataset):

    def __init__(self, year):
        super(MaskDataset, self).__init__()

        self.image_path = '../Asi_maskiert/original_image/'
        self.mask_path = '../Asi_maskiert/original_masks/'
        self.masked_images_path = '../Asi_maskiert/masked_images/'
        self.image_name = 'Image_'
        self.mask_name = 'Maske_'
        self.masked_images_name = 'tos_r8_mask_en4_'
        self.year = year

    def __getitem__(self, index):

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(self.image_path + self.image_name + self.year + '.hdf5', 'r')
        f_mask = h5py.File(self.mask_path + self.mask_name + self.year + '.hdf5', 'r')
        #f_masked_image = h5py.File(self.masked_images_path + self.masked_images_name + self.year + '.hdf5', 'r')

        #extract sst data/mask data
        image_data = f_image.get('tos_sym')
        #print(image_data)
        mask_data = f_mask.get('tos_sym')
        #masked_image_data = f_masked_image.get('tos_sym')

        #convert to pytorch tensors
        image = torch.from_numpy(image_data[index, :, :])
        mask = torch.from_numpy(mask_data[index, :, :])
        #masked_image = torch.from_numpy(masked_image_data[index, :, :])

        #image = torch.unsqueeze(image, dim=0)
        #mask = torch.unsqueeze(mask, dim=0)
        #masked_image = torch.unsqueeze(masked_image, dim=0)

        mask = mask.repeat(3, 1, 1)
        image = image.repeat(3, 1, 1)
        #masked_image = masked_image.repeat(3, 1, 1)
        #print(mask.shape)

        return mask*image, mask, image

    def __len__(self):
        
        #print length of time dimension
        f_image = h5py.File(self.image_path + self.image_name + self.year +  '.hdf5', 'r')
        image_data = f_image.get('tos_sym')
        n_samples = len(image_data[:, 1, 1])
        return n_samples

#create dataset
dataset = MaskDataset('2020')

#get sample and unpack
first_data = dataset
masked_image, mask, image = first_data[0]

#print(torch.any(image.isnan()))


