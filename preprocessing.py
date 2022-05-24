
from cmath import nan
from time import time
from matplotlib.pyplot import axis
import numpy as np
import matplotlib.pylab as plt
from sympy import N
import xarray as xr
import config as cfg
import h5py
import netCDF4
#import cdo 
#cdo = cdo.Cdo()


class preprocessing():
    
    def __init__(self, path, name, year, new_im_size, mode, depth, attribute_depth, attribute_anomaly, attribute_argo, lon1, lon2, lat1, lat2):
        super(preprocessing, self).__init__()

        self.path = path
        self.image_path = '../Asi_maskiert/pdfs/'
        self.name = name
        self.new_im_size = new_im_size
        self.mode = mode
        self.depth = depth
        self.attributes = [attribute_depth, attribute_anomaly, attribute_argo]
        self.lon1 = int(lon1)
        self.lon2 = int(lon2)
        self.lat1 = int(lat1)
        self.lat2 = int(lat2)
        self.year = year

    def __getitem__(self):
        
        ofile = self.path + self.name + self.year + '.nc'
        #ofile = self.path + self.name + self.year + '_newgrid.nc'

        #cdo.sellonlatbox(self.lon1, self.lon2, self.lat1, self.lat2, input = ifile, output = ofile)

        ds = xr.load_dataset(ofile, decode_times=False)
        
        #ds = ds.sel(lat = slice(self.lat1, self.lat2))
        #ds = ds.sel(lon = slice(self.lon1, self.lon2))


        #extract the variables from the file
        if self.mode == 'mask': 

            time_var = ds.time
            if self.attributes[2] == 'argo':
                ds = ds.sel(time=slice(200400, 202011))
            elif self.attributes[2] == 'preargo':
                ds = ds.sel(time=slice(195800, 200400))
            elif self.attributes[2] == 'full':
                ds = ds.sel(time=slice(195800, 202011))


            sst = ds.tho.values
            sst = np.where(np.isnan(sst)==False, 1, sst)
            sst = np.where(np.isnan(sst)==True, 0, sst)

        elif self.mode=='image':
            
            time_var = ds.time
            ds['time'] = netCDF4.num2date(time_var[:],time_var.units)
            if self.attributes[2] == 'argo':
                ds = ds.sel(time=slice('2004-01', '2020-10'))
            elif self.attributes[2] == 'preargo':
                ds = ds.sel(time=slice('1958-01', '2004-01'))
            elif self.attributes[2] =='full':
                ds = ds.sel(time=slice('1958-01', '2020-10'))


            ds_monthly = ds.groupby('time.month').mean('time')

            sst_mean = ds_monthly.thetao.values
            sst = ds.thetao.values

            f = h5py.File('../Asi_maskiert/original_image/baseline_climatology' + self.attributes[2] + '.hdf5', 'r')
            f.get('tos_sym')

            if self.attributes[1]=='anomalies':
                for i in range(len(sst)):
                    sst[i] = sst[i] - sst_mean[i%12]

            x = np.isnan(sst)
            sst[x] = 0

        elif self.mode=='val':

            time_var = ds.time
            if self.attributes[2] == 'argo':
                ds = ds.sel(time=slice(200400, 202012))
            elif self.attributes[2] == 'preargo':
                ds = ds.sel(time=slice(195800, 200400))
            elif self.attributes[2] == 'full':
                ds = ds.sel(time=slice(195800, 202011))


            f = h5py.File('../Asi_maskiert/original_image/baseline_climatology' + self.attributes[2] + '.hdf5', 'r')
            sst_mean = f.get('tos_sym')

            sst = ds.tho.values

            if self.attributes[1]=='anomalies':
                for i in range(len(sst)):
                    sst[i] = sst[i] - sst_mean[i%12]

            x = np.isnan(sst)
            sst[x] = 0

        n = sst.shape
        rest = np.zeros((n[0], n[1], self.new_im_size - n[2], n[3]))
        sst = np.concatenate((sst, rest), axis=2)
        n = sst.shape
        rest2 = np.zeros((n[0], n[1], n[2], self.new_im_size - n[3]))
        sst = np.concatenate((sst, rest2), axis=3)[:, :self.depth, :, :]


        #if self.attributes[1]=='depth':
        #    sst = sst[:, 0, :, :]

        n = sst.shape
        print(n)

        return sst, n


    def plot(self):
        
        sst_new, n = self.__getitem__()
        pixel_plot = plt.figure()
        pixel_plot = plt.imshow(sst_new[0, 0, :, :], vmin = -5, vmax = 5)
        plt.colorbar(pixel_plot)
        #plt.savefig(self.image_path + self.name + '.pdf')
        plt.show()

    def depths(self):

        ofile = self.path + self.name + self.year + '.nc'  
        ds = xr.load_dataset(ofile, decode_times=False)
        depth = ds.depth.values

        return depth[:self.depth]

    def save_data(self):

        sst_new, n = self.__getitem__()

        if self.mode=='val':   
            #create new h5 file with symmetric ssts
            f = h5py.File(self.path + self.name + self.year + '_' +  self.attributes[0] + '_' + self.attributes[1] + '_' + self.attributes[2] + '_' + str(cfg.in_channels) + '_observations.hdf5', 'w')

        else:
            f = h5py.File(self.path + self.name + self.year + '_' +  self.attributes[0] + '_' + self.attributes[1] + '_' + self.attributes[2] + '_' + str(cfg.in_channels) + '.hdf5', 'w')

        dset1 = f.create_dataset('tos_sym', shape=n, dtype = 'float32', data = sst_new)
        #for dim in range(0, 3):
        #    h5[cfg.data_types[0]].dims[dim].label = dname[dim]
        f.close()


cfg.set_train_args()

if cfg.mode == 'image':
    dataset = preprocessing(cfg.im_dir, cfg.im_name, cfg.im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
elif cfg.mode == 'mask':
    dataset = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.mask_year, cfg.image_size, 'mask', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
elif cfg.mode == 'both':
    dataset = preprocessing(cfg.im_dir, cfg.im_name, cfg.im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset1 = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.mask_year, cfg.image_size, 'mask', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
    dataset1.save_data()
elif cfg.mode == 'val':
    prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    prepo_obs = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.eval_mask_year, cfg.image_size, 'val', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    prepo_obs.save_data()
    prepo.save_data()
