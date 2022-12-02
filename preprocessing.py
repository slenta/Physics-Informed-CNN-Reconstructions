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
import cdo

cdo = cdo.Cdo()


class preprocessing:
    def __init__(
        self,
        path,
        name,
        year,
        new_im_size,
        mode,
        depth,
    ):
        super(preprocessing, self).__init__()

        self.path = path
        self.image_path = "../Asi_maskiert/pdfs/"
        self.name = name
        self.new_im_size = new_im_size
        self.mode = mode
        self.depth = depth
        self.year = year

    def __getitem__(self):

        ofile = self.path + self.name + self.year + ".nc"

        # if necessary: cut desired lon lat box from original image
        # ifile = ofile
        # ofile = self.path + self.name + self.year + '_newgrid.nc'
        # cdo.sellonlatbox(-65, -5, 20, 69, input = ifile, output = ofile)

        ds = xr.load_dataset(ofile, decode_times=False)

        # extract the variables from the file
        # create binary mask for training
        if self.mode == "mask":

            # adjust time dimension for training purposes
            time_var = ds.time
            if cfg.mask_argo == "argo":
                ds = ds.sel(time=slice(200400, 202011))
            elif cfg.attribute_argo == "preargo":
                ds = ds.sel(time=slice(195800, 200400))
            elif cfg.attribute_argo == "full":
                ds = ds.sel(time=slice(195800, 202011))
            if cfg.attribute_argo == "anhang":
                ds = ds.sel(time=slice(202009, 202112))

            tos = ds.tho.values
            tos = np.where(np.isnan(tos) == False, 1, 0)
            tos = tos.repeat(cfg.ensemble_member, axis=0)

        # adjust assimilation data to fit training framework
        elif self.mode == "image":

            # adjust time dimension for training purposes
            time_var = ds.time
            ds["time"] = netCDF4.num2date(time_var[:], time_var.units)
            if cfg.attribute_argo == "argo":
                ds = ds.sel(time=slice("2004-01", "2020-10"))
            elif cfg.attribute_argo == "preargo":
                ds = ds.sel(time=slice("1958-01", "2004-01"))
            elif cfg.attribute_argo == "full":
                ds = ds.sel(time=slice("1958-01", "2020-10"))
            elif cfg.attribute_argo == "anhang":
                ds = ds.sel(time=slice("2020-09", "2021-12"))

            # change total values to anomalies using calculated baseline climatology
            # (taken from ensemble mean of all assimilation data)
            f = h5py.File(
                "../Asi_maskiert/original_image/baseline_climatologyargo.hdf5",
                "r",
            )
            tos_mean = f.get("sst_mean")
            tos = ds.thetao.values

            if cfg.attribute_argo == "anhang":
                if cfg.attribute_anomaly == "anomalies":
                    for i in range(len(tos)):
                        tos[i] = tos[i] - tos_mean[(i + 9) % 12]
            else:
                if cfg.attribute_anomaly == "anomalies":
                    for i in range(len(tos)):
                        tos[i] = tos[i] - tos_mean[i % 12]

            # eliminate nan values
            tos = np.nan_to_num(tos, nan=0)

        # adjust observation data for validation purposes
        elif self.mode == "val":

            time_var = ds.time
            if cfg.attribute_argo == "argo":
                ds = ds.sel(time=slice(200400, 202011))
            elif cfg.attribute_argo == "preargo":
                ds = ds.sel(time=slice(195800, 200400))
            elif cfg.attribute_argo == "full":
                ds = ds.sel(time=slice(195800, 202011))
            if cfg.attribute_argo == "anhang":
                ds = ds.sel(time=slice(202009, 202112))

            # change total values to anomalies using calculated baseline
            # climatology (taken from ensemble mean of all assimilation data)
            f = h5py.File(
                "../Asi_maskiert/original_image/baseline_climatologyargo.hdf5",
                "r",
            )

            tos_mean = f.get("sst_mean")
            tos = ds.tho.values

            if cfg.attribute_argo == "anhang":
                if cfg.attribute_anomaly == "anomalies":
                    for i in range(len(tos)):
                        tos[i] = tos[i] - tos_mean[(i + 9) % 12]
            else:
                if cfg.attribute_anomaly == "anomalies":
                    for i in range(len(tos)):
                        tos[i] = tos[i] - tos_mean[i % 12]

            tos = np.nan_to_num(tos, nan=0)

        # create mixed dataset from observations and assimilation data to better reconstruct observational values
        elif self.mode == "mixed":

            obs_path = cfg.mask_dir + cfg.mask_name + cfg.mask_year + ".nc"
            ds_obs = xr.load_dataset(obs_path, decode_times=False)

            time_var = ds.time
            ds["time"] = netCDF4.num2date(time_var[:], time_var.units)

            if cfg.attribute_argo == "argo":
                ds = ds.sel(time=slice("2004-01", "2020-10"))
                ds_obs = ds_obs.sel(time=slice(200400, 202011))
            elif cfg.attribute_argo == "preargo":
                ds = ds.sel(time=slice("1958-01", "2004-01"))
                ds_obs = ds_obs.sel(time=slice(195800, 200400))
            elif cfg.attribute_argo == "full":
                ds = ds.sel(time=slice("1958-01", "2020-10"))
                ds_obs = ds_obs.sel(time=slice(195800, 202011))

            tos_obs = ds_obs.tho.values
            tos_im = ds.thetao.values

            tos_mixed = np.where(np.isnan(tos_obs) == True, tos_im, tos_obs)
            tos = np.array(tos_mixed)

            f = h5py.File(
                "../Asi_maskiert/original_image/baseline_climatologyargo.hdf5",
                "r",
            )
            tos_mean = f.get("sst_mean")

            if cfg.attribute_argo == "anhang":
                if cfg.attribute_anomaly == "anomalies":
                    for i in range(len(tos)):
                        tos[i] = tos[i] - tos_mean[(i + 9) % 12]
            else:
                if cfg.attribute_anomaly == "anomalies":
                    for i in range(len(tos)):
                        tos[i] = tos[i] - tos_mean[i % 12]

            tos = np.nan_to_num(tos, nan=0)

        # adjust shape of variables to fit quadratic input
        n = tos.shape
        rest = np.zeros((n[0], n[1], self.new_im_size - n[2], n[3]))
        tos = np.concatenate((tos, rest), axis=2)
        n = tos.shape
        rest2 = np.zeros((n[0], n[1], n[2], self.new_im_size - n[3]))
        tos = np.concatenate((tos, rest2), axis=3)

        # choose depth steps for multi or single layer training
        if cfg.attribute_depth == "depth":
            tos = tos[:, : self.depth, :, :]
        else:
            tos = tos[:, cfg.depth, :, :]

        # adjust shape for additional timesteps in lstm training
        if cfg.lstm_steps != 0:
            tos = np.expand_dims(tos, axis=1)
            tos = tos.repeat(cfg.lstm_steps, axis=1)
            if cfg.attribute_depth == "depth":
                for j in range(1, cfg.lstm_steps + 1):
                    for i in range(cfg.lstm_steps, tos.shape[0]):
                        tos[i, cfg.lstm_steps - j, :, :, :] = tos[
                            i - j, cfg.lstm_steps - 1, :, :, :
                        ]
            else:
                for j in range(1, cfg.lstm_steps + 1):
                    for i in range(cfg.lstm_steps, tos.shape[0]):
                        tos[i, cfg.lstm_steps - j, :, :] = tos[
                            i - j, cfg.lstm_steps - 1, :, :
                        ]
        print(tos.shape)

        return tos

    def plot(self):

        # plot variable for quick check
        tos_new = self.__getitem__()
        pixel_plot = plt.figure()
        pixel_plot = plt.imshow(tos_new[0, 0, :, :], vmin=-5, vmax=5)
        plt.colorbar(pixel_plot)
        plt.show()

    def depths(self):

        # get depth steps for later heat content calculations
        ofile = cfg.im_dir + "Image_" + cfg.im_year + ".nc"
        ds = xr.load_dataset(ofile, decode_times=False)
        depth = ds.depth.values

        return depth[: self.depth]

    def save_data(self):

        # save variable data in hdf5 format
        tos_new = self.__getitem__()

        if cfg.lstm_steps != 0:
            if self.mode == "val":
                # create new h5 file with symmetric toss
                f = h5py.File(
                    f"{self.path}{self.name}{self.year}_{cfg.attribute_depth}_{str(cfg.depth)}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}_observations.hdf5",
                    "w",
                )
            elif self.mode == "mixed":
                f = h5py.File(
                    f"{self.path}Mixed_{self.year}_{cfg.attribute_depth}_{str(cfg.depth)}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}.hdf5",
                    "w",
                )
            elif self.mode == "mask":
                f = h5py.File(
                    f"{self.path}{self.name}{self.year}_{cfg.attribute_depth}_{str(cfg.depth)}_{cfg.attribute_anomaly}_{cfg.mask_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}_{str(cfg.ensemble_member)}.hdf5",
                    "w",
                )
            else:
                f = h5py.File(
                    f"{self.path}{self.name}{self.year}_{cfg.attribute_depth}_{str(cfg.depth)}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_lstm_{str(cfg.lstm_steps)}.hdf5",
                    "w",
                )
        else:
            if self.mode == "val":
                # create new h5 file with symmetric toss
                f = h5py.File(
                    f"{self.path}{self.name}{self.year}_{cfg.attribute_depth}_{str(cfg.depth)}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}_observations.hdf5",
                    "w",
                )
            elif self.mode == "mixed":
                f = h5py.File(
                    f"{self.path}Mixed_{self.year}_{cfg.attribute_depth}_{str(cfg.depth)}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}.hdf5",
                    "w",
                )
            elif self.mode == "mask":
                f = h5py.File(
                    f"{self.path}{self.name}{self.year}_{cfg.attribute_depth}_{str(cfg.depth)}_{cfg.attribute_anomaly}_{cfg.mask_argo}_{str(cfg.in_channels)}_{str(cfg.ensemble_member)}.hdf5",
                    "w",
                )
            else:
                f = h5py.File(
                    f"{self.path}{self.name}{self.year}_{cfg.attribute_depth}_{str(cfg.depth)}_{cfg.attribute_anomaly}_{cfg.attribute_argo}_{str(cfg.in_channels)}.hdf5",
                    "w",
                )

        # create dataset with variable
        f.create_dataset("tos_sym", shape=tos_new.shape, dtype="float32", data=tos_new)
        # label dimensions
        # for dim, dname in in zip(dims, dnames):
        #    h5[cfg.data_types[0]].dim.label = dname
        f.close()


cfg.set_train_args()

if cfg.prepro_mode == "image":
    dataset = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.im_year,
        cfg.image_size,
        "image",
        cfg.in_channels,
    )
    dataset.save_data()
elif cfg.prepro_mode == "mask":
    dataset = preprocessing(
        cfg.mask_dir,
        cfg.mask_name,
        cfg.mask_year,
        cfg.image_size,
        "mask",
        cfg.in_channels,
    )
    dataset.save_data()
elif cfg.prepro_mode == "both":
    dataset = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.im_year,
        cfg.image_size,
        "image",
        cfg.in_channels,
    )
    dataset1 = preprocessing(
        cfg.mask_dir,
        cfg.mask_name,
        cfg.mask_year,
        cfg.image_size,
        "mask",
        cfg.in_channels,
    )
    dataset.save_data()
    dataset1.save_data()
elif cfg.prepro_mode == "val":
    prepro_mask = preprocessing(
        cfg.mask_dir,
        cfg.mask_name,
        cfg.mask_year,
        cfg.image_size,
        "mask",
        cfg.in_channels,
    )
    prepro = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.eval_im_year,
        cfg.image_size,
        "image",
        cfg.in_channels,
    )
    prepro_obs = preprocessing(
        cfg.mask_dir,
        cfg.mask_name,
        cfg.eval_mask_year,
        cfg.image_size,
        "val",
        cfg.in_channels,
    )
    prepro_obs.save_data()
    prepro_mask.save_data()
    prepro.save_data()
elif cfg.prepro_mode == "mixed":
    dataset = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.im_year,
        cfg.image_size,
        "mixed",
        cfg.in_channels,
    )
    dataset.save_data()
