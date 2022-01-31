from matplotlib.pyplot import plot
import pylab as plt
import numpy as np
import h5py
from preprocessing import preprocessing
import torch
from torch.utils import data
from torch.utils.data import Dataset
import xarray as xr
import config as cfg


class MaskDataset(Dataset):

    def __init__(self, mask_year, im_year, mode, depth = True, in_channels = cfg.in_channels, prepro = False, im_size = cfg.image_sizes):
        super(MaskDataset, self).__init__()

        self.image_path = '../Asi_maskiert/original_image/'
        self.mask_path = '../Asi_maskiert/original_masks/'
        self.image_name = 'Image_'
        self.mask_name = 'Maske_'
        self.im_year = im_year
        self.year = mask_year
        self.mode = mode
        self.depth = depth
        self.prepro = prepro
        self.im_size = im_size
        self.in_channels = in_channels

    def __getitem__(self, index):

        if self.prepro == True:
            dataset_i = preprocessing(self.image_path + self.image_name+ self.im_year, self.im_size, mode='image')
            dataset_m = preprocessing(self.mask_path + self.mask_name + self.year, self.im_size, mode='mask')

            dataset_i.save_data()
            dataset_m.save_data()

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(self.image_path + self.image_name  + self.im_year + '_newgrid' + '.hdf5', 'r')
        f_mask = h5py.File(self.mask_path + self.mask_name + self.year + '_newgrid' + '.hdf5', 'r')

        #extract sst data/mask data
        image = f_image.get('tos_sym')
        mask = f_mask.get('tos_sym')

        n = image.shape
        im_new = []

        if self.mode == 'train':
            for i in range(n[0]):
                if i%5 >= 1:
                    im_new.append(image[i])
        elif self.mode == 'val':
            mask = mask[:cfg.eval_timesteps]
            for i in range(n[0]):
                if i%5 == 0:
                    im_new.append(image[i])

        im_new = np.array(im_new)
        np.random.shuffle(im_new)
        np.random.shuffle(mask)

        #depth indicators
        if self.depth == False:
            im_new  = torch.from_numpy(im_new[index, 0, :, :])
            mask = torch.from_numpy(mask[index, 0, :, :])
            
            #Repeat to fit input channels
            mask = mask.repeat(self.in_channels, 1, 1)
            im_new = im_new.repeat(self.in_channels, 1, 1)
        else:
            im_new = torch.from_numpy(im_new[index, :self.in_channels, :, :])
            mask = torch.from_numpy(mask[index, :self.in_channels, :, :])

        return mask*im_new, mask, im_new, mask*im_new, mask

    def __len__(self):
        
        f_image = h5py.File(self.image_path + self.image_name + self.im_year + '.hdf5', 'r')
        image = f_image.get('tos_sym')

        n = image.shape
        im_new = []

        if self.mode == 'train':
            for i in range(n[0]):
                if i%5 >= 1:
                    im_new.append(image[i])
        elif self.mode == 'val':
            for i in range(n[0]):
                if i%5 == 0:
                    im_new.append(image[i])

        im_new = np.array(im_new)
        n = im_new.shape
        length = n[0]

        return length

    def depth(self):

        mi, ma, im_new = self.__getitem__(0)
        n = im_new.shape
        depth = n[1]

        return depth



#create dataset
#dataset = MaskDataset('1970', '3d_1958_2020', prepro=False, mode = 'val')

#get sample and unpack
#image, mask, gt, image_1, mask_1 = dataset[0]
#print(mask.shape)
