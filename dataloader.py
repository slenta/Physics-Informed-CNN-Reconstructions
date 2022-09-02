from matplotlib.pyplot import axis, plot
import matplotlib.pyplot as plt
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
        self.shuffle = shuffle
        self.im_year = im_year

    def __getitem__(self, index):

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(f'{cfg.im_dir}{cfg.im_name}{self.im_year}_{cfg.attribute_depth}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}.hdf5', 'r')
        f_mask = h5py.File(f'{cfg.mask_dir}{cfg.mask_name}{cfg.mask_year}_{cfg.attribute_depth}_{cfg.attribute_anomaly}_{cfg.mask_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}_{str(cfg.ensemble_member)}.hdf5', 'r')


        #extract sst data/mask data
        image = f_image.get('tos_sym')
        mask = f_mask.get('tos_sym')
        
        n = image.shape

        if self.shuffle == True:
            np.random.shuffle(np.array(image))
            np.random.shuffle(np.array(mask))

        if self.mode == 'test':
            mask = mask[:8]
            image = image[:8]

        #convert to pytorch tensors and adjust depth dimension
        if cfg.attribute_depth=='depth':
            mask = mask[:n[0], :, :, :]
            gt = torch.from_numpy(image[index, :self.in_channels, :, :])
            mask = torch.from_numpy(mask[index, :self.in_channels, :, :])
        
        else:
            if len(mask.shape) == 4:
                mask = mask[:n[0], :, :, :]
                mask = torch.from_numpy(mask[index, 0, :, :])
                gt = torch.from_numpy(image[index, 0, :, :])
            else:
                mask = mask[:n[0], :, :]
                mask = torch.from_numpy(mask[index, :, :])
                gt = torch.from_numpy(image[index, :, :])
            
            #Repeat to fit input channels
            mask = mask.repeat(3, 1, 1)
            gt = gt.repeat(3, 1, 1)

        if cfg.lstm_steps!=0:
            masked = mask[:, cfg.lstm_steps, :, :, :] * gt[:, cfg.lstm_steps, :, :, :]
        else:
            masked = mask * gt

        return masked, mask, gt

    def __len__(self):
        
        f_image = h5py.File(f'{cfg.im_dir}{cfg.im_name}{self.im_year}_{cfg.attribute_depth}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}.hdf5', 'r')
        image = f_image.get('tos_sym')
        n = image.shape
        gt = []

        if self.mode == 'train':
            for i in range(n[0]):
                if i%5 >= 1:
                    gt.append(image[i])
        elif self.mode == 'test':
            mask = mask[:8]
            for i in range(n[0]):
                if i%5 == 0:
                    gt.append(image[i])
            gt = gt[:8]
        elif self.mode == 'eval':
            gt = image

        gt = np.array(gt)
        length = gt.shape[0]

        return length
        


class ValDataset(Dataset):
    
    def __init__(self, im_year, mask_year, depth, in_channels):
        super(ValDataset, self).__init__()

        self.im_year = im_year
        self.mask_year = mask_year
        self.depth = depth
        self.in_channels = in_channels

    def __getitem__(self, index):

        #get h5 file for image, mask, image plus mask and define relevant variables (tos)
        f_image = h5py.File(f'{cfg.im_dir}{cfg.im_name}{self.im_year}_{cfg.attribute_depth}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}.hdf5', 'r')
        f_masked = h5py.File(f'{cfg.mask_dir}{cfg.mask_name}{self.mask_year}_{cfg.attribute_depth}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}_observations.hdf5', 'r')

        #extract sst data/mask data
        gt = np.array(f_image.get('tos_sym'))
        masked = np.array(f_masked.get('tos_sym'))
        mask = np.array(np.where(masked!=0, 1, 0))

        n = gt.shape

        #convert to pytorch tensors and adjust depth dimension
        if cfg.attribute_depth=='depth':
            mask = mask[:n[0], :, :, :]
            gt = torch.tensor(gt[index, :self.in_channels, :, :], dtype=torch.float)
            mask = torch.tensor(mask[index, :self.in_channels, :, :], dtype=torch.float)
            masked = torch.tensor(masked[index, :self.in_channels, :, :], dtype=torch.float)

        else:
            mask = mask[:n[0], :, :]
            gt = torch.tensor(gt[index, :, :], dtype=torch.float)
            mask = torch.tensor(mask[index, :, :], dtype=torch.float)
            masked = torch.tensor(masked[index, :, :], dtype=torch.float)

            mask = mask.repeat(3, 1, 1)
            gt = gt.repeat(3, 1, 1)
            masked = masked.repeat(3, 1, 1)
        
        return masked, mask, gt

    def __len__(self):
        
        f_image = h5py.File(f'{cfg.im_dir}{cfg.im_name}{self.im_year}_{cfg.attribute_depth}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}.hdf5', 'r')
        image = f_image.get('tos_sym')
        gt = np.array(image)
        
        length = gt.shape[0]

        return length
        


#dataset1 = SpecificValDataset(12*27 + 11, '11_1985')
#mi, m, i = dataset1[0]

#dataset1 = MaskDataset(7, '2020_newgrid', '3d_1958_2020_newgrid', 'train')
#mi, m, i, = dataset1[3]
#print(mi.shape, m.shape, i.shape)

#f_mask = h5py.File('../Asi_maskiert/original_masks/Maske_2004_2020.hdf5', 'r')
#mask = f_mask.get('tos_sym')
