import math
import pylab as plt
import numpy as np
import netCDF4 as nc
import h5py
import torch
from torch.utils.data import Dataset




#dataloader and dataloader

def preprocessing(path, name, type, plot):
    
    ds = nc.Dataset(path + name + '.nc')
        
    #extract the variables from the file
    lon = ds['lon'][:]
    lat = ds['lat'][:]
    time = ds['time'][:]

    if type == 'mask':
        sst = ds['sao'][:]
    elif type == 'image':
        sst = ds['tos'][:]


    #complete sst data to create symmetric shape
    rest = np.ones((754, 36, 256)) * 9999
    sst_new = np.concatenate((sst, rest), axis=1)

    #plot ssts in 2d plot
    if plot == True:
        pixel_plot = plt.figure()
        pixel_plot = plt.imshow(sst_new[1], vmin=-30, vmax=45)
        plt.colorbar(pixel_plot)
        plt.savefig('Asi_maskiert/pdfs/' + name + '.pdf')

    #create new h5 file with symmetric ssts
    f = h5py.File(path + name + '.hdf5', 'w')
    sst_new = f.create_dataset('tos_sym', (754, 256, 256))
    f.close()


class MaskDataset(Dataset):

    def __init__(self, year):
        super(MaskDataset, self).__init__()

        self.image_path = 'Asi_maskiert/original_image/'
        self.mask_path = 'Asi_maskiert/original_masks/'
        self.masked_images_path = 'Asi_maskiert/masked_images/'
        self.year = year

    def __getitem__(self, index):

        #create h5 datasets
        preprocessing(self.image_path , 'Assimilation_1958_2020', 'image', True)
        preprocessing(self.mask_path, 'Maske_' + self.year, 'mask', False)
        preprocessing(self.masked_images_path, 'tos_r8_mask_en4_' + self.year, 'image', True)

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(self.image_path + '.hdf5')
        f_mask = h5py.File(self.mask_path + self.year + '.hdf5')
        f_masked_image = h5py.File(self.masked_images_path + self.year + '.hdf5')

        #extract sst data/mask data
        image_data = f_image.get('tos_sym')
        mask_data = f_mask.get('tos_sym')
        masked_image_data = f_masked_image.get('tos_sym')

        #convert to pytorch tensors
        image = torch.from_numpy(image_data[index, :, :])
        mask = torch.from_numpy(mask_data[index, :, :])
        masked_image = torch.from_numpy(masked_image_data[index, :, :])

        return masked_image, mask, image

    def __len__(self):
        
        #print length of time dimension
        f_image = h5py.File(self.image_path + self.year + '.hdf5')
        image_data = f_image.get('tos_sym')
        n_samples = len(image_data[:, 1, 1])
        return n_samples

#create dataset
dataset = MaskDataset('2020')

#get sample and unpack
first_data = dataset[0]
time, features, labels = first_data
#print(time, features, labels)



