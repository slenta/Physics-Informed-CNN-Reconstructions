
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


            tos = ds.tho.values
            tos = np.where(np.isnan(tos)==False, 1, tos)
            tos = np.where(np.isnan(tos)==True, 0, tos)

            tos = tos.repeat(cfg.ensemble_member, axis=0)


        elif self.mode=='image':
            
            time_var = ds.time
            ds['time'] = netCDF4.num2date(time_var[:],time_var.units)
            if self.attributes[2] == 'argo':
                ds = ds.sel(time=slice('2004-01', '2020-10'))
            elif self.attributes[2] == 'preargo':
                ds = ds.sel(time=slice('1958-01', '2004-01'))
            elif self.attributes[2] =='full':
                ds = ds.sel(time=slice('1958-01', '2020-10'))



            f = h5py.File('../Asi_maskiert/original_image/baseline_climatology' + self.attributes[2] + '.hdf5', 'r')
            tos_mean = f.get('sst_mean')
            
            tos = ds.thetao.values
            
            if self.attributes[1] == 'anomalies':
                for i in range(len(tos)):
                    tos[i] = tos[i] - tos_mean[i%12]

            tos = np.nan_to_num(tos, nan=0)


        elif self.mode=='val':

            time_var = ds.time
            if self.attributes[2] == 'argo':
                ds = ds.sel(time=slice(200400, 202011))
            elif self.attributes[2] == 'preargo':
                ds = ds.sel(time=slice(195800, 200400))
            elif self.attributes[2] == 'full':
                ds = ds.sel(time=slice(195800, 202011))



            f = h5py.File('../Asi_maskiert/original_image/baseline_climatology' + self.attributes[2] + '.hdf5', 'r')
            tos_mean = f.get('sst_mean')
            tos = ds.tho.values

            if self.attributes[1]=='anomalies':
                for i in range(len(tos)):
                    tos[i] = tos[i] - tos_mean[i%12]

            tos = np.nan_to_num(tos, nan=0)

        elif self.mode == 'mixed':
            
            obs_path = cfg.mask_dir + cfg.mask_name + cfg.mask_year + '.nc'
            ds_obs = xr.load_dataset(obs_path, decode_times=False)

            time_var = ds.time
            time_obs = ds_obs.time
            ds['time'] = netCDF4.num2date(time_var[:],time_var.units)

            if self.attributes[2] == 'argo':
                ds = ds.sel(time=slice('2004-01', '2020-10'))
                ds_obs = ds_obs.sel(time=slice(200400, 202011))
            elif self.attributes[2] == 'preargo':
                ds = ds.sel(time=slice('1958-01', '2004-01'))
                ds_obs = ds_obs.sel(time=slice(195800, 200400))
            elif self.attributes[2] =='full':
                ds = ds.sel(time=slice('1958-01', '2020-10'))
                ds_obs = ds_obs.sel(time=slice(195800, 202011))


            tos_obs = ds_obs.tho.values
            tos_im = ds.thetao.values

            tos_mixed = np.where(np.isnan(tos_obs)==True, tos_im, tos_obs)
            tos = np.array(tos_mixed)

            f = h5py.File('../Asi_maskiert/original_image/baseline_climatology' + self.attributes[2] + '.hdf5', 'r')
            tos_mean = f.get('sst_mean')

            if self.attributes[1]=='anomalies':
                for i in range(len(tos)):
                    tos[i] = tos[i] - tos_mean[i%12]

            tos = np.nan_to_num(tos, nan=0)

        n = tos.shape
        rest = np.zeros((n[0], n[1], self.new_im_size - n[2], n[3]))
        tos = np.concatenate((tos, rest), axis=2)
        n = tos.shape
        rest2 = np.zeros((n[0], n[1], n[2], self.new_im_size - n[3]))
        tos = np.concatenate((tos, rest2), axis=3)

        if self.attributes[0] == 'depth':
            tos = tos[:, :self.depth, :, :]
        else:
            tos = tos[:, cfg.depth, :, :]

        if cfg.lstm_steps != 0:
            tos = np.expand_dims(tos, axis=1)
            tos = tos.repeat(cfg.lstm_steps, axis=1)
            if self.attributes[0] == 'depth':
                for j in range(1, cfg.lstm_steps + 1):
                    for i in range(cfg.lstm_steps, tos.shape[0]):
                        tos[i, cfg.lstm_steps - j, :, :, :] = tos[i - j, cfg.lstm_steps - 1, :, :, :] 
            else:
                for j in range(1, cfg.lstm_steps + 1):
                    for i in range(cfg.lstm_steps, tos.shape[0]):
                        tos[i, cfg.lstm_steps - j, :, :] = tos[i - j, cfg.lstm_steps - 1, :, :] 

        n = tos.shape
        return tos, n


    def plot(self):
        
        tos_new, n = self.__getitem__()
        pixel_plot = plt.figure()
        pixel_plot = plt.imshow(tos_new[0, 0, :, :], vmin = -5, vmax = 5)
        plt.colorbar(pixel_plot)
        #plt.savefig(self.image_path + self.name + '.pdf')
        plt.show()

    def depths(self):

        ofile = cfg.im_dir + 'Image_' + cfg.im_year + '.nc'  
        ds = xr.load_dataset(ofile, decode_times=False)
        depth = ds.depth.values

        return depth[:self.depth]

    def save_data(self):

        tos_new, n = self.__getitem__()

        if cfg.lstm_steps!=0:
            if self.mode=='val':   
                #create new h5 file with symmetric toss
                f = h5py.File(f'{self.path}{self.name}{self.year}_{self.attributes[0]}_{str(cfg.depth)}_{self.attributes[1]}_{self.attributes[2]}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}_observations.hdf5', 'w') 
            elif self.mode == 'mixed':
                f = h5py.File(f'{self.path}Mixed_{self.year}_{self.attributes[0]}_{str(cfg.depth)}_{self.attributes[1]}_{self.attributes[2]}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}.hdf5', 'w')
            elif self.mode == 'mask':
                f = h5py.File(f'{self.path}{self.name}{self.year}_{self.attributes[0]}_{str(cfg.depth)}_{self.attributes[1]}_{self.attributes[2]}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}_{str(cfg.ensemble_member)}.hdf5', 'w')
            else:
                f = h5py.File(f'{self.path}{self.name}{self.year}_{self.attributes[0]}_{str(cfg.depth)}_{self.attributes[1]}_{self.attributes[2]}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}.hdf5', 'w')
        else:
            if self.mode=='val':   
                #create new h5 file with symmetric toss
                f = h5py.File(f'{self.path}{self.name}{self.year}_{self.attributes[0]}_{str(cfg.depth)}_{self.attributes[1]}_{self.attributes[2]}_{str(cfg.in_channels)}_observations.hdf5', 'w') 
            elif self.mode == 'mixed':
                f = h5py.File(f'{self.path}Mixed_{self.year}_{self.attributes[0]}_{str(cfg.depth)}_{self.attributes[1]}_{self.attributes[2]}_{str(cfg.in_channels)}.hdf5', 'w')
            elif self.mode == 'mask':
                f = h5py.File(f'{self.path}{self.name}{self.year}_{self.attributes[0]}_{str(cfg.depth)}_{self.attributes[1]}_{self.attributes[2]}_{str(cfg.in_channels)}_{str(cfg.ensemble_member)}.hdf5', 'w')
            else:
                f = h5py.File(f'{self.path}{self.name}{self.year}_{self.attributes[0]}_{str(cfg.depth)}_{self.attributes[1]}_{self.attributes[2]}_{str(cfg.in_channels)}.hdf5', 'w')


        dset1 = f.create_dataset('tos_sym', shape=n, dtype = 'float32', data = tos_new)
        #for dim in range(0, 3):
        #    h5[cfg.data_types[0]].dims[dim].label = dname[dim]
        f.close()


cfg.set_train_args()

if cfg.prepro_mode == 'image':
    dataset = preprocessing(cfg.im_dir, cfg.im_name, cfg.im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
elif cfg.prepro_mode == 'mask':
    dataset = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.mask_year, cfg.image_size, 'mask', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
elif cfg.prepro_mode == 'both':
    dataset = preprocessing(cfg.im_dir, cfg.im_name, cfg.im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset1 = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.mask_year, cfg.image_size, 'mask', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
    dataset1.save_data()
elif cfg.prepro_mode == 'val':
    prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    prepo_obs = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.eval_mask_year, cfg.image_size, 'val', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    prepo_obs.save_data()
    prepo.save_data()
elif cfg.prepro_mode == 'mixed':
    dataset = preprocessing(cfg.im_dir, cfg.im_name, cfg.im_year, cfg.image_size, 'mixed', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()