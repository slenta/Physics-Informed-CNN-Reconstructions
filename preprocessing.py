
from time import time
import numpy as np
import matplotlib.pylab as plt
from sympy import N
import xarray as xr
import config as cfg
import h5py
import netCDF4


class preprocessing():
    
    def __init__(self, path, name, new_im_size, mode, depth, attribute1, attribute2, lon1, lon2, lat1, lat2):
        super(preprocessing, self).__init__()

        self.path = path
        self.image_path = '../Asi_maskiert/pdfs/'
        self.name = name
        self.new_im_size = new_im_size
        self.mode = mode
        self.depth = depth
        self.attributes = [attribute1, attribute2]
        self.lon1 = int(lon1)
        self.lon2 = int(lon2)
        self.lat1 = int(lat1)
        self.lat2 = int(lat2)

    def __getitem__(self):
        
        ifile = self.path + self.name + '.nc'
        ds = xr.load_dataset(ifile, decode_times=False)
        
        #ds = ds.sel(lat = slice(self.lat1, self.lat2))
        #ds = ds.sel(lon = slice(self.lon1, self.lon2))


        #extract the variables from the file
        if self.mode == 'mask': 
            sst = ds.tho.values
            x = np.isnan(sst)
            n = sst.shape
            for i in range(n[0]):
                for j in range(n[1]):
                    for k in range(n[2]):
                        for l in range(n[3]):
                            if np.isnan(sst[i, j, k, l]) == True:
                                sst[i, j, k, l] = 0
                            else:
                                sst[i, j, k, l] = 1

        elif self.mode == 'image':
            
            time_var = ds.time
            ds['time'] = netCDF4.num2date(time_var[:],time_var.units)
            ds_monthly = ds.groupby('time.month').mean('time')
            ds = ds.sel(time=slice('2004-01', '2020-10'))

            sst_mean = ds_monthly.thetao.values
            sst = ds.thetao.values

            if self.attributes[0]=='anomalies':
                for i in range(len(sst)):
                    sst[i] = sst[i] - sst_mean[i%12]

            x = np.isnan(sst)
            n = sst.shape
            sst[x] = 0

        rest = np.zeros((n[0], n[1], self.new_im_size - n[2], n[3]))
        sst = np.concatenate((sst, rest), axis=2)
        n = sst.shape
        rest2 = np.zeros((n[0], n[1], n[2], self.new_im_size - n[3]))
        sst = np.concatenate((sst, rest2), axis=3)[:, :self.depth, :, :]

        if self.attributes[1]=='depth':
            sst = sst[:, 0, :, :]

        n = sst.shape

        return sst, n


    def plot(self):
        
        sst_new, n = self.__getitem__()
        pixel_plot = plt.figure()
        pixel_plot = plt.imshow(sst_new[0, 0, :, :], vmin = -5, vmax = 5)
        print(sst_new.shape)
        plt.colorbar(pixel_plot)
        #plt.savefig(self.image_path + self.name + '.pdf')
        plt.show()

    def save_data(self):

        sst_new, n = self.__getitem__()

        #create new h5 file with symmetric ssts
        f = h5py.File(self.path + self.name + '_' +  self.attributes[0] + '_' + self.attributes[1] + '.hdf5', 'w')
        dset1 = f.create_dataset('tos_sym', shape=n, dtype = 'float32', data = sst_new)
        f.close()


cfg.set_preprocessing_args()

if cfg.mode == 'image':
    print(cfg.attribute0, cfg.attribute1)
    dataset = preprocessing(cfg.image_dir, cfg.image_name, cfg.image_size, 'image', cfg.depth, cfg.attribute0, cfg.attribute1, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
elif cfg.mode == 'mask':
    dataset = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.image_size, 'mask', cfg.depth, cfg.attributes)
    dataset.save_data()
elif cfg.mode == 'both':
    dataset = preprocessing(cfg.image_dir, cfg.image_name, cfg.image_size, 'image', cfg.depth, cfg.attributes, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset1 = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.image_size, 'mask', cfg.depth, cfg.attributes, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
    dataset1.save_data()

