from numpy.core.fromnumeric import shape
import torch
from torch.utils import data
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from image import unnormalize
import h5py
import matplotlib.pyplot as plt
import config as cfg
from numpy import ma
import sys
import os

# from utils.corr_2d_ttest import corr_2d_ttest
from collections import namedtuple

from scipy.stats import pearsonr
from preprocessing import preprocessing
import xarray as xr
import netCDF4 as nc
import cdo
import cartopy.crs as ccrs

cdo = cdo.Cdo()

sys.path.append("./")


def evaluate(
    model, dataset, device, filename, lambda_dict, criterion, writer, iteration
):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])

    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)

    image = torch.as_tensor(image).to(cfg.device)
    mask = torch.as_tensor(mask).to(cfg.device)
    gt = torch.as_tensor(gt).to(cfg.device)

    with torch.no_grad():
        output = model(image.to(device), mask.to(device))
    output_comp = mask * image + (1 - mask) * output

    # calculate loss function and apply backpropagation
    if cfg.lstm_steps != 0:
        loss_dict = criterion(
            mask[:, cfg.lstm_steps - 1, :, :, :],
            output[:, cfg.lstm_steps - 1, :, :, :],
            gt[:, cfg.lstm_steps - 1, :, :, :],
        )
    else:
        loss_dict = criterion(mask, output, gt)
    loss = 0.0

    for key, factor in lambda_dict.items():
        value = factor * loss_dict[key]
        loss += value
        writer.add_scalar("val_loss_{:s}".format(key), value.item(), iteration)

    # grid = make_grid(
    #    torch.cat((unnormalize(image), unnormalize(mask), unnormalize(output),
    #               unnormalize(output_comp), unnormalize(gt)), dim=0))

    n = image.shape

    image = np.array(image.to("cpu"))
    mask = np.array(mask.to("cpu"))
    output = np.array(output.to("cpu"))
    output_comp = np.array(output_comp.to("cpu"))
    gt = np.array(gt.to("cpu"))

    f = h5py.File(filename + ".hdf5", "w")
    dset1 = f.create_dataset("image", shape=image.shape, dtype="float32", data=image)
    dset2 = f.create_dataset("output", shape=output.shape, dtype="float32", data=output)
    dset3 = f.create_dataset(
        "output_comp", shape=output_comp.shape, dtype="float32", data=output_comp
    )
    dset4 = f.create_dataset("mask", shape=mask.shape, dtype="float32", data=mask)
    dset5 = f.create_dataset("gt", shape=gt.shape, dtype="float32", data=gt)
    f.close()

    # save_image(grid, filename + '.jpg')


def infill(model, dataset, partitions, iter, name):
    if not os.path.exists(cfg.val_dir):
        os.makedirs("{:s}".format(cfg.val_dir))
    image = []
    mask = []
    gt = []
    output = []

    if partitions > dataset.__len__():
        partitions = dataset.__len__()

    print(dataset.__len__())
    if dataset.__len__() % partitions != 0:
        print(
            "WARNING: The size of the dataset should be dividable by the number of partitions. The last "
            + str(dataset.__len__() % partitions)
            + " time steps will not be infilled."
        )
    for split in range(partitions):
        image_part, mask_part, gt_part = zip(
            *[
                dataset[i + split * (dataset.__len__() // partitions)]
                for i in range(dataset.__len__() // partitions)
            ]
        )
        image_part = torch.stack(image_part)
        mask_part = torch.stack(mask_part)
        gt_part = torch.stack(gt_part)
        # get results from trained network
        with torch.no_grad():
            output_part = model(image_part.to(cfg.device), mask_part.to(cfg.device))

        if cfg.lstm_steps == 0:
            image_part = image_part[:, :, :, :].to(torch.device("cpu"))
            mask_part = mask_part[:, :, :, :].to(torch.device("cpu"))
            gt_part = gt_part[:, :, :, :].to(torch.device("cpu"))
            output_part = output_part[:, :, :, :].to(torch.device("cpu"))

        else:
            image_part = image_part[:, cfg.lstm_steps - 1, :, :, :].to(
                torch.device("cpu")
            )
            mask_part = mask_part[:, cfg.lstm_steps - 1, :, :, :].to(
                torch.device("cpu")
            )
            gt_part = gt_part[:, cfg.lstm_steps - 1, :, :, :].to(torch.device("cpu"))
            output_part = output_part[:, cfg.lstm_steps - 1, :, :, :].to(
                torch.device("cpu")
            )

        image.append(image_part)
        mask.append(mask_part)
        gt.append(gt_part)
        output.append(output_part)

    image = torch.cat(image)
    mask = torch.cat(mask)
    gt = torch.cat(gt)
    output = torch.cat(output)

    # create output_comp
    output_comp = mask * image + (1 - mask) * output
    cvar = [gt, output, output_comp, image, mask]
    cname = ["gt", "output", "output_comp", "image", "mask"]
    dname = ["time", "lat", "lon"]

    h5 = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{iter}_{name}_{cfg.eval_im_year}.hdf5",
        "w",
    )
    for var, name in zip(cvar, cname):
        h5.create_dataset(
            name=name, shape=var.shape, dtype=float, data=var.to(torch.device("cpu"))
        )
    h5.close()

    return output, gt


def heat_content_timeseries(depth_steps, iteration, name, anomalies=""):
    rho = 1025  # density of seawater
    shc = 3850  # specific heat capacity of seawater

    if cfg.val_cut:
        cut = "_cut"
    else:
        cut = ""

    f = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{iteration}_{name}_{cfg.eval_im_year}{cut}.hdf5",
        "r",
    )
    fb = h5py.File(
        f"{cfg.im_dir}baseline_climatologyargo{cut}.hdf5",
        "r",
    )

    output = np.array(f.get("output"))
    gt = np.array(f.get("gt"))
    image = np.array(f.get("image"))
    mask = np.array(f.get("mask"))
    cvar = [gt, output, image, mask]

    if anomalies == "_full":
        thetao_mean = np.array(fb.get("sst_mean"))[:, : cfg.in_channels, :, :]
        mask = np.where(mask == 0, np.NaN, 1)
        image = image * mask

        for var in cvar:
            for i in range(var.shape[0]):
                var[i] = var[i] + thetao_mean[i % 12]

    continent_mask = np.where(gt[0, 0, :, :] == 0, np.NaN, 1)
    mask = np.where(mask == 0, np.NaN, 1)

    # take spatial mean of network output and ground truth
    masked_output = np.nansum(output * mask, axis=(3, 2))
    masked_gt = np.nansum(image * mask, axis=(3, 2))
    output = np.nansum(output * continent_mask, axis=(3, 2))
    gt = np.nansum(gt * continent_mask, axis=(3, 2))

    n = output.shape
    hc_net = np.zeros(n[0])
    hc_gt = np.zeros(n[0])
    hc_net_masked = np.zeros(n[0])
    hc_gt_masked = np.zeros(n[0])

    for i in range(n[0]):
        hc_net[i] = (
            np.sum(
                [
                    (depth_steps[k] - depth_steps[k - 1]) * output[i, k] * rho * shc
                    for k in range(1, n[1])
                ]
            )
            + depth_steps[0] * output[i, 0] * rho * shc
        )
        hc_gt[i] = (
            np.sum(
                [
                    (depth_steps[k] - depth_steps[k - 1]) * gt[i, k] * rho * shc
                    for k in range(1, n[1])
                ]
            )
            + depth_steps[0] * gt[i, 0] * rho * shc
        )
        hc_net_masked[i] = (
            np.sum(
                [
                    (depth_steps[k] - depth_steps[k - 1])
                    * masked_output[i, k]
                    * rho
                    * shc
                    for k in range(1, n[1])
                ]
            )
            + depth_steps[0] * masked_output[i, 0] * rho * shc
        )
        hc_gt_masked[i] = (
            np.sum(
                [
                    (depth_steps[k] - depth_steps[k - 1]) * masked_gt[i, k] * rho * shc
                    for k in range(1, n[1])
                ]
            )
            + depth_steps[0] * masked_gt[i, 0] * rho * shc
        )

    f_final = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/timeseries_{iteration}_{name}_{cfg.eval_im_year}{anomalies}{cut}.hdf5",
        "w",
    )

    f_final.create_dataset(name="net_ts", shape=hc_net.shape, dtype=float, data=hc_net)
    f_final.create_dataset(name="gt_ts", shape=hc_gt.shape, dtype=float, data=hc_gt)
    f_final.create_dataset(
        name="net_ts_masked", shape=hc_net_masked.shape, dtype=float, data=hc_net_masked
    )
    f_final.create_dataset(
        name="gt_ts_masked", shape=hc_gt_masked.shape, dtype=float, data=hc_gt_masked
    )
    f.close()


# simple function to plot point wise correlations between two variables (time, lon, lat)
def correlation(var_1, var_2):
    SET = namedtuple("SET", "nsim method alpha")
    n = var_1.shape
    corr = np.zeros(shape=(n[1], n[2]))
    for i in range(n[1]):
        for j in range(n[2]):
            corr[i, j] = pearsonr(var_1[:, i, j], var_2[:, i, j])[0]
    # corr, significance = corr_2d_ttest(
    #    var_1, var_2, options=SET(nsim=1000, method="ttest", alpha=0.01), nd=3
    # )
    # sig = np.where(significance == True)

    return corr


# function to calculate running standard deviation or mean
def running_mean_std(var, mode, del_t):
    n = var.shape
    var_out = np.zeros(n)
    var_out = var_out[: n[0] - (del_t - 1)]

    if mode == "mean":
        for k in range(len(var_out)):
            var_out[k] = np.nanmean(var[k : k + del_t], axis=0)
    elif mode == "std":
        for k in range(len(var_out)):
            var_out[k] = np.nanstd(var[k : k + del_t], axis=0)

    return np.array(var_out)


# calculating heat content gridpoint wise
def heat_content_single(image, depths=False, anomalies="", month=13):
    rho = 1025  # density of seawater
    shc = 3850  # specific heat capacity of seawater

    if cfg.val_cut:
        fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid_cut.hdf5", "r")
        continent_mask = fm.get("continent_mask")
        valcut = "_cut"
    else:
        fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5", "r")
        continent_mask = fm.get("continent_mask")
        valcut = ""

    n = image.shape

    prepo = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.eval_im_year,
        cfg.image_size,
        "image",
        n[1],
    )
    depth_steps = prepo.depths()
    if type(depths) != bool:
        depth_steps = depths

    if anomalies == "_full":
        fb = h5py.File(
            f"../Asi_maskiert/original_image/baseline_climatologyargo{valcut}.hdf5",
            "r",
        )
        thetao_mean = np.array(fb.get("sst_mean"))[:, : cfg.in_channels, :, :]

        if month != 13:
            image = image + thetao_mean[month]
        else:
            for i in range(n[0]):
                image[i] = image[i] + thetao_mean[i % 12]

    if cfg.val_cut:
        image = image
    else:
        image = image * continent_mask

    # take spatial mean of network output and ground truth
    if len(n) == 4:
        hc = np.zeros(shape=(n[0], n[2], n[3]))
        for i in range(n[0]):
            for j in range(n[2]):
                for l in range(n[3]):
                    hc[i, j, l] = (
                        np.nansum(
                            [
                                (depth_steps[k] - depth_steps[k - 1])
                                * image[i, k, j, l]
                                * rho
                                * shc
                                for k in range(1, n[1])
                            ]
                        )
                        + depth_steps[0] * image[i, 0, j, l] * rho * shc
                    )
    else:
        hc = np.zeros(shape=(n[1], n[2]))
        for j in range(n[1]):
            for l in range(n[2]):
                hc[j, l] = (
                    np.nansum(
                        [
                            (depth_steps[k] - depth_steps[k - 1])
                            * image[k, j, l]
                            * rho
                            * shc
                            for k in range(1, n[0])
                        ]
                    )
                    + depth_steps[0] * image[0, j, l] * rho * shc
                )

    return hc


# calculating heat content gridpoint wise
def heat_content(depth_steps, iteration, name, anomalies=""):
    rho = 1025  # density of seawater
    shc = 3850  # specific heat capacity of seawater

    if cfg.val_cut:
        cut = "_cut"
    else:
        cut = ""

    f = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{iteration}_{name}_{cfg.eval_im_year}{cut}.hdf5",
        "r",
    )

    output = np.array(f.get("output"))
    gt = np.array(f.get("gt"))
    image = np.array(f.get("image"))
    mask = np.array(f.get("mask"))
    mask = np.where(mask == 0, np.NaN, 1)

    fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5", "r")
    fb = h5py.File(
        f"../Asi_maskiert/original_image/baseline_climatologyargo{cut}.hdf5",
        "r",
    )

    continent_mask = fm.get("continent_mask")
    continent_mask = np.where(gt[0, 0, :, :] == 0, np.NaN, 1)

    if anomalies == "_full":
        thetao_mean = np.array(fb.get("sst_mean"))[:, : cfg.in_channels, :, :]
        cvar = [gt, output, image, mask]

        for var in cvar:
            for i in range(var.shape[0]):
                var[i] = var[i] + thetao_mean[i % 12]

    # take spatial mean of network output and ground truth
    masked_output = output * mask
    masked_gt = image * mask
    output = output * continent_mask
    gt = gt * continent_mask

    n = output.shape
    hc_net = np.zeros(shape=(n[0], n[2], n[3]))
    hc_gt = np.zeros(shape=(n[0], n[2], n[3]))
    hc_net_masked = np.zeros(shape=(n[0], n[2], n[3]))
    hc_gt_masked = np.zeros(shape=(n[0], n[2], n[3]))

    for i in range(n[0]):
        for j in range(n[2]):
            for l in range(n[3]):
                hc_net[i, j, l] = (
                    np.sum(
                        [
                            (depth_steps[k] - depth_steps[k - 1])
                            * output[i, k, j, l]
                            * rho
                            * shc
                            for k in range(1, n[1])
                        ]
                    )
                    + depth_steps[0] * output[i, 0, j, l] * rho * shc
                )
                hc_gt[i, j, l] = (
                    np.sum(
                        [
                            (depth_steps[k] - depth_steps[k - 1])
                            * gt[i, k, j, l]
                            * rho
                            * shc
                            for k in range(1, n[1])
                        ]
                    )
                    + depth_steps[0] * gt[i, 0, j, l] * rho * shc
                )
                hc_net_masked[i, j, l] = (
                    np.sum(
                        [
                            (depth_steps[k] - depth_steps[k - 1])
                            * masked_output[i, k, j, l]
                            * rho
                            * shc
                            for k in range(1, n[1])
                        ]
                    )
                    + depth_steps[0] * masked_output[i, 0, j, l] * rho * shc
                )
                hc_gt_masked[i, j, l] = (
                    np.sum(
                        [
                            (depth_steps[k] - depth_steps[k - 1])
                            * masked_gt[i, k, j, l]
                            * rho
                            * shc
                            for k in range(1, n[1])
                        ]
                    )
                    + depth_steps[0] * masked_gt[i, 0, j, l] * rho * shc
                )

    final_vars = [hc_net, hc_gt, hc_net_masked, hc_gt_masked]
    final_names = ["hc_net", "hc_gt", "hc_net_masked", "hc_gt_masked"]

    f_final = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/heatcontent_{iteration}_{name}_{cfg.eval_im_year}{cut}{anomalies}.hdf5",
        "w",
    )
    for nam, var in zip(final_names, final_vars):
        f_final.create_dataset(name=nam, shape=var.shape, data=var)
    f.close()


def create_snapshot_image(model, dataset, filename):
    image, mask, gt = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    image = torch.stack(image).to(cfg.device)
    mask = torch.stack(mask).to(cfg.device)
    gt = torch.stack(gt).to(cfg.device)

    with torch.no_grad():
        output = model(image, mask)

    if cfg.lstm_steps == 0:
        image = image[:, :, :, :].to(torch.device("cpu"))
        gt = gt[:, :, :, :].to(torch.device("cpu"))
        mask = mask[:, :, :, :].to(torch.device("cpu"))
        output = output[:, :, :, :].to(torch.device("cpu"))

    else:
        # select last element of lstm sequence as evaluation element
        image = image[:, cfg.lstm_steps - 1, :, :, :].to(torch.device("cpu"))
        gt = gt[:, cfg.lstm_steps - 1, :, :, :].to(torch.device("cpu"))
        mask = mask[:, cfg.lstm_steps - 1, :, :, :].to(torch.device("cpu"))
        output = output[:, cfg.lstm_steps - 1, :, :, :].to(torch.device("cpu"))

    output_comp = mask * image + (1 - mask) * output

    # set mask
    mask = 1 - mask
    image = ma.masked_array(image, mask)

    mask = ma.masked_array(mask, mask)

    for c in range(gt.shape[1]):
        if cfg.attribute_anomaly == "anomalies":
            vmin, vmax = (-5, 5)
        else:
            vmin, vmax = (-10, 35)
        data_list = [
            image[:, c, :, :],
            mask[:, c, :, :],
            output[:, c, :, :],
            output_comp[:, c, :, :],
            gt[:, c, :, :],
        ]

        # plot and save data
        fig, axes = plt.subplots(
            nrows=len(data_list), ncols=image.shape[0], figsize=(20, 20)
        )
        fig.patch.set_facecolor("black")
        for i in range(len(data_list)):
            for j in range(image.shape[0]):
                axes[i, j].axis("off")
                axes[i, j].imshow(np.squeeze(data_list[i][j]), vmin=vmin, vmax=vmax)
        plt.subplots_adjust(wspace=0.012, hspace=0.012)
        plt.savefig(f"{filename}_{str(c)}.jpg", bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close("all")


def area_cutting_nc(mode):
    f = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{cfg.resume_iter}_{mode}_{cfg.eval_im_year}.hdf5",
        "r",
    )

    ds_compare = xr.load_dataset(f"{cfg.im_dir}Image_r9_full_newgrid.nc")

    # load all variables from file
    cname = ["gt", "output", "image", "mask"]
    gt, output, image, mask = [f.get(name) for name in cname]
    cvar = [gt, output, image, mask]

    # cut out additional input from nn
    image = image[:, :, :107, :124]
    mask = mask[:, :, :107, :124]
    gt = gt[:, :, :107, :124]
    output = output[:, :, :107, :124]

    # read out coordinates from compare file
    lat = ds_compare.lat.values
    lon = ds_compare.lon.values
    time = ds_compare.time.values[: gt.shape[0]]

    depth = ds_compare.depth.values[: gt.shape[1]]
    print(image.shape)

    ds = xr.Dataset(
        data_vars=dict(
            image=(["time", "depth", "y", "x"], image),
            mask=(["time", "depth", "y", "x"], mask),
            gt=(["time", "depth", "y", "x"], gt),
            output=(["time", "depth", "y", "x"], output),
        ),
        coords=dict(
            time=(["time"], time),
            depth=(["depth"], depth),
            lon=(["y", "x"], lon),
            lat=(["y", "x"], lat),
        ),
        attrs=dict(description=f"NA cut of {mode} infilled"),
    )

    # assign regular lonlat coordinates
    lon = ds_compare.lon
    lat = ds_compare.lat
    depth = ds_compare.depth

    depth_sub = depth.where(depth.isin(ds.depth)).dropna(dim="depth")
    ds = ds.assign_coords({"lon": lon, "lat": lat, "depth": depth_sub})

    # save as nc and cut with cdo
    ifile = f"{cfg.im_dir}{cfg.resume_iter}_{mode}_{cfg.eval_im_year}.nc"
    ds.to_netcdf(ifile)
    ofile = f"{cfg.im_dir}{cfg.resume_iter}_{mode}_{cfg.eval_im_year}_cut.nc"
    cdo.sellonlatbox(cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2, input=ifile, output=ofile)

    # load cut variable from nc out file
    cname_new = ["gt_new", "output_new", "image_new", "mask_new"]
    ds_out = xr.load_dataset(ofile)
    gt_new, output_new, image_new, mask_new = (
        ds_out.gt.values,
        ds_out.output.values,
        ds_out.image.values,
        ds_out.mask.values,
    )
    cvar_new = [gt_new, output_new, image_new, mask_new]

    # save all variables in hdf5 File
    h5 = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{cfg.resume_iter}_{mode}_{cfg.eval_im_year}_cut.hdf5",
        "w",
    )
    for name, varnew in zip(cname, cvar_new):
        h5.create_dataset(
            name=name,
            shape=varnew.shape,
            dtype=float,
            data=varnew,
        )
    h5.close()

    # remove created temporary files
    os.remove(ifile)
    os.remove(ofile)

    return gt_new, output_new, image_new, mask_new


def area_cutting_single(var):
    ds_compare = xr.load_dataset(f"{cfg.im_dir}Image_r9_full_newgrid.nc")

    lat = ds_compare.lat.values
    lon = ds_compare.lon.values
    time = ds_compare.time.values[: var.shape[0]]

    if len(var.shape) == 4:
        var = var[:, :, :107, :124]

        depth = ds_compare.depth.values[: var.shape[1]]

        ds = xr.Dataset(
            data_vars=dict(
                variable=(["time", "depth", "y", "x"], var),
            ),
            coords=dict(
                time=(["time"], time),
                depth=(["depth"], depth),
                lon=(["y", "x"], lon),
                lat=(["y", "x"], lat),
            ),
            attrs=dict(description=f"NA cut of variable"),
        )

        # assign regular lonlat coordinates
        lon = ds_compare.lon
        lat = ds_compare.lat
        depth = ds_compare.depth
        depth_sub = depth.where(depth.isin(ds.depth)).dropna(dim="depth")
        ds = ds.assign_coords({"lon": lon, "lat": lat, "depth": depth_sub})

    elif len(var.shape) == 3:
        var = var[:, :107, :124]

        ds = xr.Dataset(
            data_vars=dict(
                variable=(["time", "y", "x"], var),
            ),
            coords=dict(
                time=(["time"], time),
                lon=(["y", "x"], lon),
                lat=(["y", "x"], lat),
            ),
            attrs=dict(description=f"NA cut of variable"),
        )

        # assign regular lonlat coordinates
        lon = ds_compare.lon
        lat = ds_compare.lat
        ds = ds.assign_coords({"lon": lon, "lat": lat})

    elif len(var.shape) == 2:
        var = var[:107, :124]

        ds = xr.Dataset(
            data_vars=dict(
                variable=(["y", "x"], var),
            ),
            coords=dict(
                lon=(["y", "x"], lon),
                lat=(["y", "x"], lat),
            ),
            attrs=dict(description=f"NA cut of variable"),
        )

        # assign regular lonlat coordinates
        lon = ds_compare.lon
        lat = ds_compare.lat
        ds = ds.assign_coords({"lon": lon, "lat": lat})

    # save as nc and cut with cdo
    ifile = f"{cfg.im_dir}ac_tmp.nc"
    ds.to_netcdf(ifile)
    ofile = f"{cfg.im_dir}ac_tmp_cut.nc"
    cdo.sellonlatbox(cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2, input=ifile, output=ofile)

    # load cut variable from nc out file
    ds_out = xr.load_dataset(ofile)
    var_new = ds_out.variable.values

    # remove created temporary files
    os.remove(ifile)
    os.remove(ofile)

    return var_new


def area_cutting(mode, depth):
    ds_compare = xr.load_dataset(f"{cfg.im_dir}Image_r9.nc")
    ifile = f"{cfg.im_dir}Image_r9.nc"
    ofile = f"{cfg.im_dir}Image_r9_newgrid.nc"

    # cdo.sellonlatbox(-65, -5, 20, 69, input = ifile, output = ofile)

    ds_compare = xr.load_dataset(f"{cfg.im_dir}Image_r9_full_newgrid.nc")

    if cfg.attribute_argo == "anhang":
        length = 764
    else:
        length = 752

    lat = np.array(ds_compare.lat.values)
    lon = np.array(ds_compare.lon.values)
    time = np.array(ds_compare.time.values)[:length]

    lon_out = np.arange(cfg.lon1, cfg.lon2)
    lat_out = np.arange(cfg.lat1, cfg.lat2)

    f = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{cfg.resume_iter}_{mode}_{cfg.eval_im_year}.hdf5",
        "r",
    )

    cname = ["gt", "output", "image", "mask"]
    cname_new = ["gt_new", "output_new", "image_new", "mask_new"]

    for name, newname in zip(cname, cname_new):
        globals()[name] = np.array([f.get(name)]).squeeze(axis=0)
        globals()[newname] = np.zeros(
            shape=(len(time), depth, len(lat_out), len(lon_out)), dtype="float32"
        )

    for la in lat_out:
        for lo in lon_out:
            x_lon, y_lon = np.where(np.round(lon) == lo)
            x_lat, y_lat = np.where(np.round(lat) == la)
            x_out = []
            y_out = []
            for x, y in zip(x_lon, y_lon):
                for a, b in zip(x_lat, y_lat):
                    if (x, y) == (a, b):
                        x_out.append(x)
                        y_out.append(y)
            for i in range(len(time)):
                for j in range(depth):
                    for name, newname in zip(cname, cname_new):
                        globals()[newname][
                            i, j, la - min(lat_out), lo - min(lon_out)
                        ] = np.mean(
                            [globals()[name][i, j, x, y] for x, y in zip(x_out, y_out)]
                        )

    for newname in cname_new:
        globals()[newname] = globals()[newname][:, :, ::-1, :]

    h5 = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{cfg.resume_iter}_{mode}_{cfg.eval_im_year}_cut.hdf5",
        "w",
    )
    for newname, name in zip(cname_new, cname):
        h5.create_dataset(
            name=name,
            shape=globals()[newname].shape,
            dtype=float,
            data=globals()[newname],
        )
    h5.close()


def create_dataset(var, val_cut):
    ds_compare = xr.load_dataset(f"{cfg.im_dir}Image_r9_full_newgrid{val_cut}.nc")

    lon = ds_compare.lon.values
    lat = ds_compare.lat.values
    depth = ds_compare.depth.values
    time = ds_compare.time.values

    n = var.shape
    time = time[: n[0]]
    if len(var.shape) == 4:
        var = var[:, :, :107, :124]
        depth = depth[: n[1]]

        ds = xr.Dataset(
            data_vars=dict(
                variable=(["time", "depth", "y", "x"], var),
            ),
            coords=dict(
                time=(["time"], time),
                depth=(["depth"], depth),
                lon=(["y", "x"], lon),
                lat=(["y", "x"], lat),
            ),
            attrs=dict(description=f"NA cut of variable"),
        )

        # assign regular lonlat coordinates
        lon = ds_compare.lon
        lat = ds_compare.lat
        depth = ds_compare.depth
        depth_sub = depth.where(depth.isin(ds.depth)).dropna(dim="depth")
        ds = ds.assign_coords({"lon": lon, "lat": lat, "depth": depth_sub})

    elif len(var.shape) == 3:
        var = var[:, :107, :124]
        ds = xr.Dataset(
            data_vars=dict(
                variable=(["time", "y", "x"], var),
            ),
            coords=dict(
                time=(["time"], time),
                lon=(["y", "x"], lon),
                lat=(["y", "x"], lat),
            ),
            attrs=dict(description=f"NA cut of variable"),
        )

        # assign regular lonlat coordinates
        lon = ds_compare.lon
        lat = ds_compare.lat
        ds = ds.assign_coords({"lon": lon, "lat": lat})

    elif len(var.shape) == 2:
        var = var[:107, :124]
        ds = xr.Dataset(
            data_vars=dict(
                variable=(["y", "x"], var),
            ),
            coords=dict(
                lon=(["y", "x"], lon),
                lat=(["y", "x"], lat),
            ),
            attrs=dict(description=f"NA cut of variable"),
        )

        # assign regular lonlat coordinates
        lon = ds_compare.lon
        lat = ds_compare.lat
        ds = ds.assign_coords({"lon": lon, "lat": lat})

    return ds


def pattern_correlation(image_1, image_2):
    image_1 = np.array(image_1)
    image_2 = np.array(image_2)

    image_1 = image_1.flatten("C")
    image_2 = image_2.flatten("C")
    corr = pearsonr(image_1, image_2)[0]

    return corr


def pattern_corr_timeseries(name, del_t=1, anomalies=""):
    if cfg.val_cut:
        cut = "_cut"
    else:
        cut = ""

    if cfg.nw_corner:
        nw = "_nw"
    else:
        nw = ""

    f = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/heatcontent_{str(cfg.resume_iter)}_{name}_{cfg.eval_im_year}{cut}{anomalies}.hdf5",
        "r",
    )

    output = np.nan_to_num(np.array(f.get("hc_net")), nan=0)
    gt = np.nan_to_num(np.array(f.get("hc_gt")), nan=0)

    if del_t != 1:
        output = running_mean_std(output, mode="mean", del_t=del_t)
        gt = running_mean_std(gt, mode="mean", del_t=del_t)

    n = output.shape

    corr_ts = np.zeros((n[0]))

    for i in range(n[0]):
        corr_ts[i] = pattern_correlation(output[i, :, :], gt[i, :, :])

    f_final = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/pattern_corr_ts_{str(cfg.resume_iter)}_{name}_{cfg.eval_im_year}_mean_{del_t}{cut}{nw}{anomalies}.hdf5",
        "w",
    )

    f_final.create_dataset(
        name="corr_ts", shape=corr_ts.shape, dtype=float, data=corr_ts
    )
    f.close()


def combine_layers(parts, anomalies=True):
    fb = h5py.File(
        f"{cfg.im_dir}baseline_climatologyargo.hdf5",
        "r",
    )

    thetao_mean = np.array(fb.get("sst_mean")[:, :20, :, :])

    names = ["output", "image", "mask", "gt"]

    if cfg.attribute_argo == "anhang":
        length = 764
    else:
        length = 752
    if cfg.nw_corner:
        nw = "_nw"
    else:
        nw = ""

    for name in names:
        globals()[f"{name}_full"] = np.zeros(shape=(length, 20, 128, 128))
        globals()[f"{name}_obs_full"] = np.zeros(shape=(length, 20, 128, 128))

    print(cfg.resume_iter)
    for depth, part in zip(range(len(parts)), parts):
        f = h5py.File(
            f"{cfg.val_dir}part_{str(part)}/validation_{str(cfg.resume_iter)}_assimilation_{cfg.mask_argo}{nw}_{cfg.eval_im_year}.hdf5",
            "r",
        )
        fo = h5py.File(
            f"{cfg.val_dir}part_{str(part)}/validation_{str(cfg.resume_iter)}_observations_{cfg.mask_argo}{nw}_{cfg.eval_im_year}.hdf5",
            "r",
        )

        for name in names:
            globals()[f"{name}"] = np.array(f.get(name))
            globals()[f"{name}_obs"] = np.array(fo.get(name))

            globals()[f"{name}_full"][:, depth, :, :] = globals()[f"{name}"][:, 0, :, :]
            globals()[f"{name}_obs_full"][:, depth, :, :] = globals()[f"{name}_obs"][
                :, 0, :, :
            ]

    if anomalies == False:
        for name in names:
            for i in range(globals()[f"{name}"].shape[0]):
                globals()[f"{name}_full"][i, :, :, :] = (
                    globals()[f"{name}_full"][i, :, :, :] + thetao_mean[i % 12, :, :, :]
                )
                globals()[f"{name}_obs_full"][i, :, :, :] = (
                    globals()[f"{name}_obs_full"][i, :, :, :]
                    + thetao_mean[i % 12, :, :, :]
                )
        ano = "_full"
    else:
        ano = ""

    f_a = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{str(cfg.resume_iter)}_assimilation_{cfg.mask_argo}{nw}_{cfg.eval_im_year}{ano}.hdf5",
        "w",
    )
    f_o = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_{str(cfg.resume_iter)}_observations_{cfg.mask_argo}{nw}_{cfg.eval_im_year}{ano}.hdf5",
        "w",
    )
    for name in names:
        f_a.create_dataset(
            name=name,
            shape=globals()[f"{name}_full"].shape,
            data=globals()[f"{name}_full"],
        )
        f_o.create_dataset(
            name=name,
            shape=globals()[f"{name}_obs_full"].shape,
            data=globals()[f"{name}_obs_full"],
        )
    f_a.close()
    f_o.close()


def ml_ensemble_iteration(members, part, iteration, length=754):
    hc_all_a, hc_all_o = np.zeros(shape=(2, members, length))
    members = np.arange(1, members + 1)
    print(members)

    for member in members:
        it = iteration + 500 * member
        if cfg.val_cut:
            file_a = f"{cfg.val_dir}{part}/timeseries_{it}_assimilation_anhang_r2_full_newgrid_cut.hdf5"
            file_o = f"{cfg.val_dir}{part}/timeseries_{it}_observations_anhang_r2_full_newgrid_cut.hdf5"
        else:
            file_a = f"{cfg.val_dir}{part}/timeseries_{it}_assimilation_anhang_r2_full_newgrid.hdf5"
            file_o = f"{cfg.val_dir}{part}/timeseries_{it}_observations_anhang_r2_full_newgrid.hdf5"

        f_a = h5py.File(file_a, "r")
        f_o = h5py.File(file_o, "r")

        hc_a = np.array(f_a.get("net_ts"))
        hc_o = np.array(f_o.get("net_ts"))

        hc_all_a[member - 1, :] = hc_a
        hc_all_o[member - 1, :] = hc_o

    hc_all_a = np.array(hc_all_a)
    hc_all_o = np.array(hc_all_o)

    return hc_all_a, hc_all_o


def hc_ml_ensemble(members, part, iteration, length=754):
    hc_all_a, hc_all_o = np.zeros(shape=(2, members, length))
    members = np.arange(1, members + 1)

    for member in members:
        if cfg.val_cut:
            file_a = f"{cfg.val_dir}{part}/timeseries_{iteration}_assimilation_anhang_r{member}_full_newgrid_cut.hdf5"
            file_o = f"{cfg.val_dir}{part}/timeseries_{iteration}_observations_anhang_r{member}_full_newgrid_cut.hdf5"
        else:
            file_a = f"{cfg.val_dir}{part}/timeseries_{iteration}_assimilation_anhang_r{member}_full_newgrid.hdf5"
            file_o = f"{cfg.val_dir}{part}/timeseries_{iteration}_observations_anhang_r{member}_full_newgrid.hdf5"

        f_a = h5py.File(file_a, "r")
        f_o = h5py.File(file_o, "r")

        hc_a = np.array(f_a.get("net_ts"))
        hc_o = np.array(f_o.get("net_ts"))

        hc_all_a[member - 1, :] = hc_a
        hc_all_o[member - 1, :] = hc_o

    hc_all_a = np.array(hc_all_a)
    hc_all_o = np.array(hc_all_o)

    return hc_all_a, hc_all_o


def hc_ensemble_mean_std(path=cfg.im_dir, name=cfg.im_name, members=15, length=767):
    hc_all, hc_cut_all = np.zeros(shape=(2, members, length))

    for i in range(1, members + 1):
        file = f"{path}/{name}{i}_anomalies_depth_anhang_20.hdf5"
        f = h5py.File(file, "r")
        hc = f.get("hc")
        hc_cut = f.get("hc_cut")

        hc_all[i - 1] = np.array(hc)
        hc_cut_all[i - 1] = np.array(hc_cut)

    mean = np.mean(hc_all, axis=0)
    mean_cut = np.mean(hc_cut_all, axis=0)
    std = np.std(hc_all, axis=0)
    std_cut = np.std(hc_cut_all, axis=0)

    if cfg.val_cut:
        return mean_cut, std_cut, hc_cut_all
    else:
        return mean, std, hc_all


def sst_bias_maps(start=0, end=120, sst="HadIsst", name=""):
    if cfg.val_cut:
        val_cut = "_cut"
    else:
        val_cut = ""

    # load network output and assimilation data, full data for comparison with HadIsst
    f = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_550000_observations_anhang_{cfg.eval_im_year}_full.hdf5"
    )
    fa = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/validation_550000_assimilation_anhang_{cfg.eval_im_year}_full.hdf5"
    )
    f_cm = h5py.File(f"{cfg.mask_dir}Kontinent_newgrid.hdf5")
    f_spg = h5py.File(f"{cfg.mask_dir}SPG_Maske.hdf5")

    output = np.array(f.get("output"))[:, 0, :107, :124]
    assi = np.array(f.get("gt"))[:, 0, :107, :124]

    # load sst reference data
    if sst == "HadIsst":
        ds = xr.load_dataset(f"{cfg.im_dir}HadIsst{val_cut}.nc")
        obs = ds.sst.values
    elif sst == "obs":
        obs = np.array(f.get("image"))[:, 0, :107, :124]

    # create means of early times of biased assimilation
    net_1 = output[start:end, :, :]
    assi_1 = assi[start:end, :, :]
    sst_1 = obs[start:end, :, :]

    # create mean fields at comparatively well observed times
    start_2 = 552
    end_2 = 754
    net_2 = output[start_2:end_2, :, :]
    assi_2 = assi[start_2:end_2, :, :]
    sst_2 = obs[start_2:end_2, :, :]

    # create bias fields and convert to datasets
    net_bias_1 = np.nanmean(net_1 - sst_1, axis=0)
    assi_bias_1 = np.nanmean(assi_1 - sst_1, axis=0)
    net_bias_2 = np.nanmean(net_2 - sst_2, axis=0)
    assi_bias_2 = np.nanmean(assi_2 - sst_2, axis=0)

    nb1 = create_dataset(net_bias_1, val_cut="")
    ab1 = create_dataset(assi_bias_1, val_cut="")
    nb2 = create_dataset(net_bias_2, val_cut="")
    ab2 = create_dataset(assi_bias_2, val_cut="")

    # create additional cut dataset for SPG visualization, draw in nw corner region
    north_nw = 20
    south_nw = 37
    west_nw = 25
    east_nw = 35

    nb1_cut = create_dataset(area_cutting_single(net_bias_1), val_cut="_cut")
    # nb1_cut.variable[north_nw:south_nw, west_nw] = 200
    # nb1_cut.variable[north_nw:south_nw, east_nw] = 200
    # nb1_cut.variable[north_nw, west_nw:east_nw] = 200
    # nb1_cut.variable[south_nw, west_nw : (east_nw + 1)] = 200
    ab1_cut = create_dataset(area_cutting_single(assi_bias_1), val_cut="_cut")
    # ab1_cut.variable[north_nw:south_nw, west_nw] = 200
    # ab1_cut.variable[north_nw:south_nw, east_nw] = 200
    # ab1_cut.variable[north_nw, west_nw:east_nw] = 200
    # ab1_cut.variable[south_nw, west_nw : (east_nw + 1)] = 200
    nb2_cut = create_dataset(area_cutting_single(net_bias_2), val_cut="_cut")
    # nb2_cut.variable[north_nw:south_nw, west_nw] = 200
    # nb2_cut.variable[north_nw:south_nw, east_nw] = 200
    # nb2_cut.variable[north_nw, west_nw:east_nw] = 200
    # nb2_cut.variable[south_nw, west_nw : (east_nw + 1)] = 200
    ab2_cut = create_dataset(area_cutting_single(assi_bias_2), val_cut="_cut")
    # ab2_cut.variable[north_nw:south_nw, west_nw] = 200
    # ab2_cut.variable[north_nw:south_nw, east_nw] = 200
    # ab2_cut.variable[north_nw, west_nw:east_nw] = 200
    # ab2_cut.variable[south_nw, west_nw : (east_nw + 1)] = 200

    # define mins and max for colorbar and images
    cmin = -2
    cmax = 2
    ylim = [38, 72]
    xlim = [-75, -5]

    time = (end - start) // 12
    End = 1958 + time
    cmap_1 = plt.cm.get_cmap("coolwarm").copy()

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    fig.suptitle("NA SST Bias", fontweight="bold", fontsize=15)
    ax1 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax1.set_global()
    nb1.variable.plot.pcolormesh(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=cmin,
        vmax=cmax,
        alpha=0.4,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    nb1_cut.variable.plot.pcolormesh(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap=cmap_1,
        vmin=cmin,
        vmax=cmax,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    ax1.coastlines()
    ax1.set_title(f"Neural Network SST bias January 1958 -- January {End}")
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    gls1 = ax1.gridlines(color="lightgrey", linestyle="-", draw_labels=True)
    gls1.top_labels = False  # suppress top labels
    gls1.right_labels = False  # suppress right labels
    gls1.bottom_labels = False  # suppress bottom labels
    gls1.left_labels = False  # suppress left labels
    ax2 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax2.set_global()
    ab1.variable.plot.pcolormesh(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=cmin,
        vmax=cmax,
        alpha=0.4,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    im = ab1_cut.variable.plot.pcolormesh(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap=cmap_1,
        vmin=cmin,
        vmax=cmax,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    ax2.coastlines()
    ax2.set_title(f"Assimilation SST bias January 1958 -- January {End}")
    ax2.set_ylim(ylim)
    ax2.set_xlim(xlim)
    gls2 = ax2.gridlines(color="lightgrey", linestyle="-", draw_labels=True)
    gls2.top_labels = False  # suppress top labels
    gls2.right_labels = False  # suppress right labels
    gls2.bottom_labels = False  # suppress bottom labels
    gls2.left_labels = False  # suppress left labels
    cbar = plt.colorbar(im, shrink=0.8, location="bottom")
    cbar.set_label("SST bias in °C")
    ax3 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax3.set_global()
    nb2.variable.plot.pcolormesh(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=cmin,
        vmax=cmax,
        alpha=0.4,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    nb2_cut.variable.plot.pcolormesh(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        cmap=cmap_1,
        vmin=cmin,
        vmax=cmax,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    ax3.coastlines()
    ax3.set_title("Neural Network SST bias January 2010 -- January 2020")
    ax3.set_ylim(ylim)
    ax3.set_xlim(xlim)
    gls3 = ax3.gridlines(color="lightgrey", linestyle="-", draw_labels=True)
    gls3.top_labels = False  # suppress top labels
    gls3.right_labels = False  # suppress right labels
    gls3.bottom_labels = False  # suppress bottom labels
    gls3.left_labels = False  # suppress left labels
    ax4 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax4.set_global()
    ab2.variable.plot.pcolormesh(
        ax=ax4,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=cmin,
        vmax=cmax,
        alpha=0.4,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    im = ab2_cut.variable.plot.pcolormesh(
        ax=ax4,
        transform=ccrs.PlateCarree(),
        cmap=cmap_1,
        vmin=cmin,
        vmax=cmax,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    ax4.coastlines()
    ax4.set_title("Assimilation SST bias January 2010 -- January 2020")
    ax4.set_ylim(ylim)
    ax4.set_xlim(xlim)
    gls4 = ax4.gridlines(color="lightgrey", linestyle="-", draw_labels=True)
    gls4.top_labels = False  # suppress top labels
    gls4.right_labels = False  # suppress right labels
    gls4.bottom_labels = False  # suppress bottom labels
    gls4.left_labels = False  # suppress left labels
    cbar = plt.colorbar(im, shrink=0.8, location="bottom")
    cbar.set_label("SST bias in °C")
    plt.savefig(
        f"../Asi_maskiert/pdfs/validation/part_19/nw_images/sst_biases_{sst}{name}.pdf"
    )
    plt.show()


# cfg.set_train_args()

# prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
# prepo.save_data()
# depths = prepo.depths()

# heat_content_timeseries_general(depths, cfg.eval_im_year)
