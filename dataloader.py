from matplotlib.pyplot import axis, plot
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
from torchvision.utils import make_grid
from torchvision.utils import save_image
from image import unnormalize
import config as cfg




#dataloader and dataloader
class MaskDataset(Dataset):

    def __init__(self, im_year, depth, in_channels, mode, shuffle = True):
        super(MaskDataset, self).__init__()

        self.mode = mode
        self.in_channels = in_channels
        self.depth = depth
        self.shuffle = shuffle
        self.im_year = im_year

    def __getitem__(self, index):

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(cfg.im_dir + cfg.im_name + self.im_year + '_' +  cfg.attribute_depth + '_' + cfg.attribute_anomaly + '_' + cfg.attribute_argo + '.hdf5', 'r')
        f_mask = h5py.File(cfg.mask_dir + cfg.mask_name + cfg.mask_year + '_' +  cfg.attribute_depth + '_' + cfg.attribute_anomaly + '_' + cfg.attribute_argo + '.hdf5', 'r')

        #extract sst data/mask data
        image = f_image.get('tos_sym')
        mask = f_mask.get('tos_sym')
        mask = np.repeat(mask, 5, axis=0)
        
        n = image.shape
        mask = mask[:n[0], :, :, :]
        m = mask.shape

        im_new = []

        if self.mode == 'train':
            for i in range(n[0]):
                if i%5 >= 1:
                    im_new.append(image[i])
        elif self.mode == 'test':
            mask = mask[:8]
            for i in range(n[0]):
                if i%5 == 0:
                    im_new.append(image[i])
            im_new = im_new[:8]
        elif self.mode == 'val':
            im_new = image

        im_new = np.array(im_new)

        if self.shuffle == True:
            np.random.shuffle(im_new)
        
        np.random.shuffle(mask)

        #convert to pytorch tensors
        if self.depth==True:
            im_new = torch.from_numpy(im_new[index, :self.in_channels, :, :])
            mask = torch.from_numpy(mask[index, :self.in_channels, :, :])
        
        elif self.depth ==False:
            if len(mask.shape) == 4:
                mask = torch.from_numpy(mask[index, 0, :, :])
                im_new = torch.from_numpy(im_new[index, 0, :, :])
            else:
                mask = torch.from_numpy(mask[index, :, :])
                im_new = torch.from_numpy(im_new[index, :, :])
            
            #Repeat to fit input channels
            mask = mask.repeat(3, 1, 1)
            im_new = im_new.repeat(3, 1, 1)
	    		
        return mask*im_new, mask, im_new, mask*im_new, mask

    def __len__(self):
        
        mi, ma, im_new, mis, mas = self.__getitem__(0)
        n = im_new.shape
        length = n[0]

        return length
        


class SpecificValDataset():
    
    def __init__(self, timestep, year):
        super(SpecificValDataset, self).__init__()

        self.image_path = '../Asi_maskiert/original_image/'
        self.mask_path = '../Asi_maskiert/original_masks/'
        self.image_name = 'Observation_'
        self.mask_name = 'Observation_'
        self.year = year
        self.timestep = timestep

    def __getitem__(self, index):

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(self.image_path + self.image_name + self.year + '.hdf5', 'r')
        f_mask = h5py.File(self.mask_path + self.mask_name + self.year + '.hdf5', 'r')
        f_gt = h5py.File(self.image_path + 'Image_2020.hdf5', 'r')

        #extract sst data/mask data
        image = f_image.get('tos_sym')
        mask = f_mask.get('tos_sym')
        gt = f_gt.get('tos_sym')[self.timestep, :, :]

        #repeat to insert time dimension
        image = np.repeat(image, 16, axis=0)
        mask = np.repeat(mask, 16, axis=0)
        gt = np.expand_dims(gt, axis=0)
        gt = np.repeat(gt, 16, axis=0)

        #convert to pytorch tensors
        im_new = torch.from_numpy(image[index, :, :])
        mask = torch.from_numpy(mask[index, :, :])
        gt = torch.from_numpy(gt[index, :, :])

        #bring into right shape   
        mask = mask.repeat(3, 1, 1)
        im_new = im_new.repeat(3, 1, 1)
        gt = gt.repeat(3, 1, 1)


        return im_new, mask, gt



#dataset1 = SpecificValDataset(12*27 + 11, '11_1985')
#mi, m, i = dataset1[0]

#dataset1 = MaskDataset(7, '2020_newgrid', '3d_1958_2020_newgrid', 'train')
#mi, m, i, = dataset1[3]
#print(mi.shape, m.shape, i.shape)

#f_mask = h5py.File('../Asi_maskiert/original_masks/Maske_2004_2020.hdf5', 'r')
#mask = f_mask.get('tos_sym')
