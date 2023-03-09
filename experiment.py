# creating timeseries from all assimilation ensemble members and ensemble mean

import matplotlib
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import h5py
import config as cfg
import xarray as xr
import evaluation_og as evalu
import cdo
import os
from preprocessing import preprocessing
from scipy.stats import pearsonr, norm

cdo = cdo.Cdo()

cfg.set_train_args()


###### Mask plots

# depth = 20
# part = "part_16"
# iteration = 1000000
#
# fa = h5py.File(
#    f"../Asi_maskiert/results/validation/{part}/validation_{iteration}_assimilation_full.hdf5",
#    "r",
# )
# fo = h5py.File(
#    f"../Asi_maskiert/results/validation/{part}/validation_{iteration}_observations_full.hdf5",
#    "r",
# )
# fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5", "r")
#
# continent_mask = np.array(fm.get("continent_mask"))
#
# mask = np.array(fa.get("mask")[:, 0, :, :]) * continent_mask
# image_o = np.array(fo.get("image")[:, 0, :, :]) * continent_mask
#
# mask_grey = np.where(mask == 0, np.NaN, mask) * continent_mask
# save_dir = f"../Asi_maskiert/pdfs/validation/{part}/"
#
# if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
#
# fig = plt.figure(figsize=(10, 6), constrained_layout=True)
# fig.suptitle("Anomaly North Atlantic Heat Content")
# plt.subplot(1, 2, 1)
# plt.title(f"Preargo Observations: January 1958")
# current_cmap = plt.cm.coolwarm
# current_cmap.set_bad(color="gray")
# plt.imshow(
#    image_o[0] * mask_grey[0],
#    cmap=current_cmap,
#    vmin=-3,
#    vmax=3,
#    aspect="auto",
#    interpolation=None,
# )
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
## plt.colorbar(label='Temperature in °C')
# plt.subplot(1, 2, 2)
# plt.title(f"Argo Observations: August 2020")
# im2 = plt.imshow(
#    image_o[751] * mask_grey[751],
#    cmap="coolwarm",
#    vmin=-3,
#    vmax=3,
#    aspect="auto",
#    interpolation=None,
# )
# plt.colorbar()
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# fig.savefig(
#    f"../Asi_maskiert/pdfs/validation/{part}/Masks_{iteration}.pdf", dpi=fig.dpi
# )
# plt.show()
#
#
#### Assimilation Ensemble Spread

# gt_mean, std_gt, hc_all = evalu.hc_ensemble_mean_std(
#    cfg.im_dir, name="Image_r", members=16
# )
# length = hc_all.shape[1]
# ticks = np.arange(0, length, 12 * 5)
# labels = np.arange(1958, 2021, 5)

# plt.figure(figsize=(10, 6))
# for i in range(16):
#    globals()[f"r{i}"] = hc_all[i, :]
#    globals()[f"r{i}"] = evalu.running_mean_std(
#        globals()[f"r{i}"], mode="mean", del_t=12
#    )
#    plt.plot(globals()[f"r{i}"])
# plt.xticks(ticks=ticks, labels=labels)
# plt.xlim(0, len(gt_mean))
# plt.title("Assimilation Ensemble Members: NA OHC")
# plt.grid()
# plt.ylabel("Heat Content [J]")
# plt.xlabel("Time [years]")
# plt.savefig(f"../Asi_maskiert/pdfs/validation/part_16/Ensemble_spread.pdf")
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.fill_between(
#    range(len(gt_mean)),
#    gt_mean + 2 * std_gt,
#    gt_mean - 2 * std_gt,
#    label="Ensemble Spread Reconstruction",
#    color="lightcoral",
# )
# for i in range(16):
#     globals()[f"r{i}"] = hc_all[i, :]
#     globals()[f"r{i}"] = evalu.running_mean_std(
#         globals()[f"r{i}"], mode="mean", del_t=12
#     )
#     plt.plot(globals()[f"r{i}"], alpha=0.5)
# plt.plot(
#     evalu.running_mean_std(gt_mean, mode="mean", del_t=12), color="darkred", linewidth=2
# )
# plt.xticks(ticks=ticks, labels=labels)
# plt.xlim(0, len(gt_mean))
# plt.title("Assimilation Ensemble NA OHC")
# plt.grid()
# plt.ylabel("Heat Content [J]")
# plt.xlabel("Time [years]")
# plt.savefig(f"../Asi_maskiert/pdfs/validation/part_19/Ensemble_mean.pdf")
# plt.show()

######## Uncertainty schlauch + Vorhersage argo masken

# save_part = "part_15"
# if cfg.val_cut:
#    cut = "_cut"
# else:
#    cut = ""
#
# member = "r7"
#
# f = h5py.File(
#    f"{cfg.val_dir}{save_part}/timeseries_{cfg.resume_iter}_assimilation_full_{member}_newgrid{cut}.hdf5",
#    "r",
# )
# fo = h5py.File(
#    f"{cfg.val_dir}{save_part}/timeseries_{cfg.resume_iter}_observations_full_{member}_newgrid{cut}.hdf5",
#    "r",
# )
#
# f_anhang = h5py.File(
#    f"{cfg.val_dir}{save_part}/timeseries_{cfg.resume_iter}_assimilation_anhang_{member}_anhang_newgrid{cut}.hdf5",
#    "r",
# )
# fo_anhang = h5py.File(
#    f"{cfg.val_dir}{save_part}/timeseries_{cfg.resume_iter}_observations_anhang_{member}_anhang_newgrid{cut}.hdf5",
#    "r",
# )
#
# gt_mean, std_gt, hc_all = evalu.hc_ensemble_mean_std(
#    cfg.im_dir, name="Image_r", members=16
# )
#
#
# hc_a = np.array(f.get("net_ts"))[552:]
# hc_gt = np.array(f.get("gt_ts"))[552:]
# hc_o = np.array(fo.get("net_ts"))[552:]
#
# hc_a_an = np.array(f_anhang.get("net_ts"))
# hc_gt_an = np.array(f_anhang.get("gt_ts"))
# hc_o_an = np.array(fo_anhang.get("net_ts"))
#
# hc_all_a, hc_all_o = evalu.hc_ml_ensemble(15, length=752)
# del_a = np.nanstd(hc_all_a[:, 540:], axis=0)
# del_o = np.nanstd(hc_all_o[:, 540:], axis=0)
#
# hc_gt = np.concatenate((hc_gt, hc_gt_an))
# hc_a = np.concatenate((hc_a, hc_a_an))
# hc_o = np.concatenate((hc_o, hc_o_an))
#
# print(hc_gt.shape)
#
# length = len(hc_a)
# ticks = np.arange(0, length, 12 * 2)
# labels = np.arange(2004, 2022, 2)
#
# plt.figure(figsize=(10, 6))
# plt.fill_between(
#    range(len(hc_a)),
#    hc_a + del_a,
#    hc_a - del_a,
#    label="Standard deviation Indirect Reconstruction",
#    color="lightblue",
# )
# plt.fill_between(
#    range(len(hc_gt)),
#    hc_o + del_o,
#    hc_o - del_o,
#    label="Standard deviation Direct Reconstruction",
#    color="limegreen",
# )
# plt.fill_between(
#    range(len(hc_gt)),
#    hc_gt + std_gt[542:],
#    hc_gt - std_gt[542:],
#    label="Standard deviation Assimilation",
#    color="lightcoral",
# )
# plt.plot(hc_gt, label="Assimilation Heat Content", color="darkred")
# plt.plot(
#    hc_a, label="Network Reconstructed Heat Content Assimilation", color="midnightblue"
# )
# plt.plot(
#    hc_o, label="Network Reconstructed Heat Content Observations", color="darkgreen"
# )
#
# plt.xticks(ticks=ticks, labels=labels)
# plt.title("Reconstructed Observations: Argo Era")
# plt.grid()
# plt.ylabel("Heat Content [J]")
# plt.xlabel("Time [years]")
# plt.legend()
# plt.savefig(
#    f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/Observations_timeseries_predictions.pdf"
# )
# plt.show()


# ############ Correlation plotting
#
# f_a = h5py.File(
#     f"../Asi_maskiert/results/validation/{cfg.save_part}/validation_{cfg.resume_iter}_assimilation_full.hdf5",
#     "r",
# )
# f_ah = h5py.File(
#     f"../Asi_maskiert/results/validation/{cfg.save_part}/heatcontent_{cfg.resume_iter}_assimilation_full.hdf5",
#     "r",
# )
#
# fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5", "r")
# continent_mask = np.array(fm.get("continent_mask"))
#
# hc_a = np.array(f_ah.get("net_ts")) * continent_mask
# hc_gt = np.array(f_ah.get("gt_ts")) * continent_mask
#
# correlation_argo_a, sig_argo_a = evalu.correlation(hc_a[552:], hc_gt[552:])
# correlation_preargo_a, sig_preargo_a = evalu.correlation(hc_a[:552], hc_gt[:552])
#
# acc_mean_argo_a = pearsonr(
#     np.nanmean(np.nanmean(hc_gt[552:], axis=1), axis=1),
#     np.nanmean(np.nanmean(hc_a[552:], axis=1), axis=1),
# )[0]
# acc_mean_preargo_a = pearsonr(
#     np.nanmean(np.nanmean(hc_gt[:552], axis=1), axis=1),
#     np.nanmean(np.nanmean(hc_a[:552], axis=1), axis=1),
# )[0]
#
# fig = plt.figure(figsize=(10, 6), constrained_layout=True)
# fig.suptitle("Anomaly North Atlantic Heat Content")
# plt.subplot(1, 2, 1)
# plt.title(f"Argo Reconstruction Correlation: {acc_mean_argo_a:.2f}")
# current_cmap = plt.cm.coolwarm
# current_cmap.set_bad(color="gray")
# plt.scatter(sig_argo_a[1], sig_argo_a[0], c="black", s=0.7, marker=".", alpha=0.5)
# im3 = plt.imshow(correlation_argo_a, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.subplot(1, 2, 2)
# plt.title(f"Preargo Reconstruction Correlation: {acc_mean_preargo_a:.2f}")
# current_cmap = plt.cm.coolwarm
# current_cmap.set_bad(color="gray")
# plt.scatter(sig_preargo_a[1], sig_preargo_a[0], c="black", s=0.7, marker=".", alpha=0.5)
# im3 = plt.imshow(correlation_preargo_a, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(label="Annomaly Correlation")
# fig.savefig(
#     f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/correlation_preargo_argo.pdf",
#     dpi=fig.dpi,
# )
# plt.show()
#
#
# ####################pdfs argo/preargo
# del_t = 1
# argo = cfg.mask_argo
# n_windows = 1
#
# f = h5py.File(
#     f"{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_assimilation_{argo}.hdf5",
#     "r",
# )
# fo = h5py.File(
#     f"{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_observations_{argo}.hdf5",
#     "r",
# )
#
# hc_assi = np.array(f.get("net_ts"))
# hc_gt = np.array(f.get("gt_ts"))
# hc_obs = np.array(fo.get("net_ts"))
#
# # calculate running mean, if necessary
# hc_assi = evalu.running_mean_std(hc_assi, mode="mean", del_t=del_t)
# hc_gt = evalu.running_mean_std(hc_gt, mode="mean", del_t=del_t)
# hc_obs = evalu.running_mean_std(hc_obs, mode="mean", del_t=del_t)
# len_w = len(hc_assi) // n_windows
# for i in range(n_windows):
#     globals()[f"hc_a_{str(i)}"] = hc_assi[len_w * i : len_w * (i + 1)]
#     globals()[f"hc_o_{str(i)}"] = hc_obs[len_w * i : len_w * (i + 1)]
#     globals()[f"hc_gt_{str(i)}"] = hc_gt[len_w * i : len_w * (i + 1)]
#
#     globals()[f"error_a_{str(i)}"] = np.sqrt(
#         (globals()[f"hc_a_{str(i)}"] - globals()[f"hc_gt_{str(i)}"]) ** 2
#     )
#     globals()[f"error_o_{str(i)}"] = np.sqrt(
#         (globals()[f"hc_o_{str(i)}"] - globals()[f"hc_gt_{str(i)}"]) ** 2
#     )
#
#     globals()[f"pdf_a_{str(i)}"] = norm.pdf(
#         np.sort(globals()[f"error_a_{str(i)}"]),
#         np.mean(globals()[f"error_a_{str(i)}"]),
#         np.std(globals()[f"error_a_{str(i)}"]),
#     )
#     globals()[f"pdf_o_{str(i)}"] = norm.pdf(
#         np.sort(globals()[f"error_o_{str(i)}"]),
#         np.mean(globals()[f"error_o_{str(i)}"]),
#         np.std(globals()[f"error_o_{str(i)}"]),
#     )
#
# plt.title("Error PDFs")
# if n_windows != 1:
#     for i in range(n_windows):
#         plt.plot(
#             np.sort(globals()[f"error_a_{str(i)}"]),
#             globals()[f"pdf_a_{str(i)}"],
#             label=f"Assimilation Reconstruction Error Pdf {str(i)}",
#         )
#         # plt.plot(np.sort(globals()[f'error_o_{str(i)}']), globals()[f'pdf_o_{str(i)}'], label=f'Observations Reconstruction Error Pdf {str(i)}')
# plt.grid()
# plt.ylabel("Probability Density")
# plt.xlabel("Absolute Error of Reconstruction")
# plt.legend()
# plt.savefig(f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/error_pdfs.pdf")
# plt.show()


# Calculating ML ensemble spread
# names = ["gt", "a", "o", "gt_cut", "a_cut", "o_cut"]
# for name in names:
#    globals()[f"hc_{name}_all"] = np.zeros(shape=(15, 752))
#
# for member in range(1, 16):
#
#    resume_iter = 1000000 + member * 1000
#    val_file_a = h5py.File(
#        f"{cfg.val_dir}{cfg.save_part}/timeseries_{resume_iter}_assimilation_{cfg.mask_argo}.hdf5",
#        "r",
#    )
#    val_file_o = h5py.File(
#        f"{cfg.val_dir}{cfg.save_part}/timeseries_{resume_iter}_observations_{cfg.mask_argo}.hdf5",
#        "r",
#    )
#    val_file_a_cut = h5py.File(
#        f"{cfg.val_dir}{cfg.save_part}/timeseries_{resume_iter}_assimilation_{cfg.mask_argo}_cut.hdf5",
#        "r",
#    )
#    val_file_o_cut = h5py.File(
#        f"{cfg.val_dir}{cfg.save_part}/timeseries_{resume_iter}_observations_{cfg.mask_argo}_cut.hdf5",
#        "r",
#    )
#
#    hc_gt = np.array(val_file_a.get("hc_gt"))
#    hc_a = np.array(val_file_a.get("hc_net"))
#    hc_o = np.array(val_file_o.get("hc_net"))
#    hc_gt_cut = np.array(val_file_a_cut.get("hc_gt"))
#    hc_a_cut = np.array(val_file_a_cut.get("hc_net"))
#    hc_o_cut = np.array(val_file_o_cut.get("hc_net"))
#
#    for name in names:
#        globals()[f"hc_{name}_all"][member - 1, :] = globals()[f"hc_{name}"]
#
#    # save both full and val_cut
#    h5_name = (
#        f"../Asi_maskiert/original_image/Image_r{member}_anomalies_depth_full_20.hdf5"
#    )
#    f = h5py.File(h5_name, "w")
#    for name in names:
#        f.create_dataset(
#            f"hc_{name}_all",
#            shape=globals()[f"hc_{name}_all"].shape,
#            data=globals()[f"hc_{name}_all"],
#        )
#    f.close()

##Calculate Assimilation Ensemble Spread
# depth = cfg.in_channels
# argo = "anhang"
# for member in range(1, 17):
#    print(member)
#
#    # ifile = f"/work/uo1075/decadal_system_mpi-esm-lr_enkf/data/MPI-ESM1-2-LR/asSEIKERAf/Omon/thetao/r{member}i8p4/thetao_Omon_MPI-ESM-LR_asSEIKERAf_r{member}i8p4_195801-202010.nc"
#    ofile = f"../Asi_maskiert/original_image/Image_r{member}_full_newgrid.nc"
#
#    # cdo.sellonlatbox(-65, -5, 20, 69, input=ifile, output=ofile)
#
#    ds = xr.load_dataset(ofile, decode_times=False)
#
#    f = h5py.File("../Asi_maskiert/original_image/baseline_climatologyargo.hdf5", "r")
#    tos_mean = f.get("sst_mean_newgrid")
#
#    tos = ds.thetao.values
#
#    for i in range(len(tos)):
#        tos[i] = tos[i] - tos_mean[i % 12]
#
#    tos = np.nan_to_num(tos, nan=0)
#    tos = tos[:, :depth, :, :]
#    n = tos.shape
#
#    # val_cut
#    tos_cut = evalu.area_cutting_single(tos)
#
#    prepo = preprocessing(
#        cfg.im_dir,
#        cfg.im_name,
#        cfg.eval_im_year,
#        cfg.image_size,
#        "image",
#        cfg.in_channels,
#    )
#    depth_steps = prepo.depths()
#
#    continent_mask_cut = np.where(tos_cut[0, 0, :, :] == 0, np.nan, 1)
#    continent_mask = np.where(tos[0, 0, :, :] == 0, np.nan, 1)
#
#    tos_new = np.nansum(tos_cut * continent_mask_cut, axis=(3, 2))
#    tos = np.nansum(tos * continent_mask, axis=(3, 2))
#
#    hc_cut = np.zeros(n[0])
#    hc = np.zeros(n[0])
#    rho = 1025  # density of seawater
#    shc = 3850  # specific heat capacity of seawater
#
#    for j in range(n[0]):
#        hc[j] = (
#            np.sum(
#                [
#                    (depth_steps[k] - depth_steps[k - 1]) * tos[j, k] * rho * shc
#                    for k in range(1, n[1])
#                ]
#            )
#            + depth_steps[0] * tos[j, 0] * rho * shc
#        )
#        hc_cut[j] = (
#            np.sum(
#                [
#                    (depth_steps[k] - depth_steps[k - 1]) * tos_new[j, k] * rho * shc
#                    for k in range(1, n[1])
#                ]
#            )
#            + depth_steps[0] * tos_new[j, 0] * rho * shc
#        )
#
#    # save both full and val_cut
#    h5_name = (
#        f"../Asi_maskiert/original_image/Image_r{member}_anomalies_depth_{argo}_20.hdf5"
#    )
#    f = h5py.File(h5_name, "w")
#    f.create_dataset("hc", shape=hc.shape, data=hc)
#    f.create_dataset("hc_cut", shape=hc_cut.shape, data=hc_cut)
#    f.close()

###save OHC climatology

# f = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo.hdf5",
#    "r",
# )
#
# gt = np.array(f.get("sst_mean"))
# n = gt.shape
# rho = 1025  # density of seawater
# shc = 3850  # specific heat capacity of seawater
#
# prepo = preprocessing(
#    cfg.im_dir,
#    cfg.im_name,
#    cfg.eval_im_year,
#    cfg.image_size,
#    "image",
#    40,
# )
# depth_steps = prepo.depths()
# hc_gt = np.zeros(shape=(n[0], n[2], n[3]))
#
# for i in range(n[0]):
#    for j in range(n[2]):
#        for l in range(n[3]):
#            hc_gt[i, j, l] = (
#                np.sum(
#                    [
#                        (depth_steps[k] - depth_steps[k - 1])
#                        * gt[i, k, j, l]
#                        * rho
#                        * shc
#                        for k in range(1, n[1])
#                    ]
#                )
#                + depth_steps[0] * gt[i, 0, j, l] * rho * shc
#            )
#
# print(hc_gt.shape)
#
#
# f = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo_hc.hdf5",
#    "w",
# )
# f.create_dataset(name="hc", shape=hc_gt.shape, data=hc_gt)
# f.close()

###cut climatology to SPG
# f = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo_hc.hdf5",
#    "r",
# )
# hc = f.get("hc")
# print(hc.shape)
#
# ds_compare = xr.load_dataset(f"{cfg.im_dir}Image_r9_newgrid.nc")
# depth = 40
# length = 12
#
# lat = np.array(ds_compare.lat.values)
# lon = np.array(ds_compare.lon.values)
# time = np.array(ds_compare.time.values)[:length]
#
# lon_out = np.arange(cfg.lon1, cfg.lon2)
# lat_out = np.arange(cfg.lat1, cfg.lat2)
#
# hc_new = np.zeros(shape=(len(time), len(lat_out), len(lon_out)), dtype="float32")
#
# for la in lat_out:
#    for lo in lon_out:
#        x_lon, y_lon = np.where(np.round(lon) == lo)
#        x_lat, y_lat = np.where(np.round(lat) == la)
#        x_out = []
#        y_out = []
#        for x, y in zip(x_lon, y_lon):
#            for a, b in zip(x_lat, y_lat):
#                if (x, y) == (a, b):
#                    x_out.append(x)
#                    y_out.append(y)
#        for i in range(len(time)):
#            hc_new[i, la - min(lat_out), lo - min(lon_out)] = np.mean(
#                [hc[i, x, y] for x, y in zip(x_out, y_out)]
#            )
#
# hc_new = hc_new[:, ::-1, :]
#
#
# f = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo_hc_cut.hdf5",
#    "w",
# )
# f.create_dataset(name="hc_new", shape=hc_new.shape, data=hc_new)
# f.close()


#### get IAP OHC into correct shape and substract anomalies
# ifile = f"/work/uo1075/u241265/obs/ohc/IAP_ohc700m_mm_1960_2016.nc"
# ofile = f"{cfg.im_dir}/IAP_ohc700m_mm_1960_2016_cut.nc"
## cdo.sellonlatbox(cfg.lon1, cfg.lon2 - 1, cfg.lat1, cfg.lat2, input=ifile, output=ofile)
# ds = xr.open_dataset(ofile, decode_times=False)
#
# hc = ds["heatcontent"]
# print(hc.shape)
# print(hc[20])
#
# f = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo_hc_cut.hdf5",
#    "r",
# )
# hc_mean = f.get("hc_new")
# print(hc_mean[10])
#
# for i in range(len(hc)):
#    hc[i] = hc[i] - hc_mean[(i) % 12]
#
# f = h5py.File(
#    "../Asi_maskiert/original_image/iap_hc_cut.hdf5",
#    "w",
# )
# f.create_dataset(name="hc", shape=hc.shape, data=hc)
# f.close()


###### Show SPG on NA map

# ds = xr.load_dataset(f"{cfg.im_dir}Image_r9_newgrid.nc")
# fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5", "r")
# continent_mask = np.array(fm.get("continent_mask"))
#
# lat = np.array(ds.lat.values)
# lon = np.array(ds.lon.values)
#
# lon_out = np.arange(cfg.lon1, cfg.lon2)
# lat_out = np.arange(45, 60)
#
# tos = np.array(ds.thetao.values[0, 0, :, :])
# n = tos.shape
# rest = np.zeros((128 - n[0], n[1])) * np.nan
# tos = np.concatenate((tos, rest), axis=0)
# n = tos.shape
# rest2 = np.zeros((n[0], 128 - n[1])) * np.nan
# tos = np.concatenate((tos, rest2), axis=1)
# n = tos.shape
# tos_new = tos.copy()
# x_out = []
# y_out = []
#
# for la in lat_out:
#    for lo in lon_out:
#        x_lon, y_lon = np.where(np.round(lon) == lo)
#        x_lat, y_lat = np.where(np.round(lat) == la)
#        for x, y in zip(x_lon, y_lon):
#            if x in x_lat and y in y_lat:
#                if x not in x_out:
#                    x_out.append(x)
#                if y not in y_out:
#                    y_out.append(y)
#
# for x in range(n[0]):
#    for y in range(n[1]):
#        if x not in x_out or y not in y_out:
#            tos[x, y] = np.nan
#
# tos = np.where(np.isnan(tos) == False, 1, tos)
#
# for x in range(n[0] - 1):
#    for y in range(n[1] - 1):
#        if tos[x, y] != 1:
#            if (
#                tos[x + 1, y] != 1
#                and tos[x - 1, y] != 1
#                and tos[x, y - 1] != 1
#                and tos[x, y + 1] != 1
#            ):
#                tos[x, y] = 50
#            else:
#                tos[x, y] = np.nan
#
#
# SPG_lines = np.where(np.isnan(tos) == False, 1, np.nan)
#
# cmap_1 = plt.cm.get_cmap("coolwarm").copy()
# cmap_1.set_bad(color="grey")
# cmap_2 = plt.cm.get_cmap("jet").copy()
# cmap_2.set_bad(color="black")
#
# plt.title("Subpolar Gyre Region")
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.imshow(tos_new, cmap=cmap_1, vmin=-3, vmax=60)
# plt.imshow(SPG_lines, cmap=cmap_2, vmin=-3, vmax=60, alpha=0.3)
# plt.savefig(
#    f"../Asi_maskiert/pdfs/validation/part_16/SPG_grid.pdf",
# )
# plt.show()
#
# path = f"{cfg.mask_dir}SPG_Maske.hdf5"
# fs = h5py.File(path, "w")
# fs.create_dataset(name="SPG", shape=SPG_lines.shape, data=SPG_lines)
# fs.close()

### Pattern Correlation for anhang


# f_a = h5py.File(
#    f"../Asi_maskiert/results/validation/{cfg.save_part}/validation_{cfg.resume_iter}_assimilation_full_{cfg.eval_im_year}_newgrid_cut.hdf5",
#    "r",
# )
#
# f_an = h5py.File(
#    f"../Asi_maskiert/results/validation/{cfg.save_part}/validation_{cfg.resume_iter}_assimilation_anhang_{cfg.eval_im_year}_anhang_newgrid_cut.hdf5",
#    "r",
# )
#
# f_o = h5py.File(
#    f"../Asi_maskiert/results/validation/{cfg.save_part}/validation_{cfg.resume_iter}_observations_full_{cfg.eval_im_year}_newgrid_cut.hdf5",
#    "r",
# )
#
# f_on = h5py.File(
#    f"../Asi_maskiert/results/validation/{cfg.save_part}/validation_{cfg.resume_iter}_observations_anhang_{cfg.eval_im_year}_anhang_newgrid_cut.hdf5",
#    "r",
# )
#
# a = np.array(f_a.get("output"))[552:]
# gt = np.array(f_a.get("gt"))[552:]
# o = np.array(f_o.get("output"))[552:]
#
# a_an = np.array(f_an.get("output"))
# o_an = np.array(f_on.get("output"))
# gt_an = np.array(f_an.get("gt"))
#
# a_full = np.concatenate((a, a_an), axis=0)
# o_full = np.concatenate((o, o_an), axis=0)
# gt_full = np.concatenate((gt, gt_an), axis=0)
#
# a_hc = evalu.heat_content_single(a_full)
# o_hc = evalu.heat_content_single(o_full)
# gt_hc = evalu.heat_content_single(gt_full)
#
# n = a_hc.shape
# corr_o = []
# corr_a = []
# for i in range(n[0]):
#    a_flat = a_hc[i, :, :].flatten()
#    o_flat = o_hc[i, :, :].flatten()
#    gt_flat = gt_hc[i, :, :].flatten()
#
#    corr_o.append(pearsonr(o_flat, gt_flat)[0])
#    corr_a.append(pearsonr(a_flat, gt_flat)[0])
#
# length = len(corr_o)
# ticks = np.arange(0, length, 12 * 5)
# labels = np.arange(2004, 2022, 5)
#
# plt.figure(figsize=(10, 6))
# plt.plot(corr_o)
# plt.plot(corr_a)
# plt.grid()
# plt.legend()
# plt.ylim(0, 1)
# plt.xticks(ticks=ticks, labels=labels)
# plt.title("OHC Pattern Correlation: Monthly means")
# plt.xlabel("Time in years")
# plt.ylabel(f"Pattern Correlation as ACC")
# plt.savefig(
#    f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/pattern_cor_timeseries_{str(cfg.resume_iter)}_{cfg.eval_im_year}_{cfg.attribute_argo}_complete.pdf"
# )
# plt.show()

###### Drawing coastlines


# fc = h5py.File(
#    f"{cfg.val_dir}part_16/validation_1000000_assimilation_full_cut.hdf5",
#    "r",
# )
# gt = np.array(fc.get("gt"))[0, 0, :, :]
# continent_mask = np.where(gt == 0, np.nan, 1)
#
# n = continent_mask.shape
#
# for x in range(n[0] - 1):
#    for y in range(n[1] - 1):
#        if continent_mask[x, y] != 1:
#            if (
#                continent_mask[x + 1, y] != 1
#                and continent_mask[x - 1, y] != 1
#                and continent_mask[x, y - 1] != 1
#                and continent_mask[x, y + 1] != 1
#            ):
#                continent_mask[x, y] = np.nan
#            else:
#                continent_mask[x, y] = 1e6
#
#
# cmap_1 = plt.cm.get_cmap("coolwarm").copy()
# cmap_1.set_bad(color="grey")
# cmap_2 = plt.cm.get_cmap("coolwarm").copy()
# cmap_2.set_bad(color="black")
#
#
# coastlines = np.where(continent_mask == 1e6, np.nan, 1)
# continent_mask = np.where(continent_mask == 1e6, np.nan, continent_mask)
#
# plt.imshow(gt * continent_mask, cmap=cmap_1, vmin=-3, vmax=3)
# plt.imshow(coastlines * gt, cmap=cmap_2, vmin=-3, vmax=3, alpha=0.5)
# plt.show()
#
# path = f"{cfg.mask_dir}Kontinent_newgrid_cut.hdf5"
# f = h5py.File(path, "w")
# f.create_dataset("continent_mask", shape=continent_mask.shape, data=continent_mask)
# f.create_dataset("coastlines", shape=coastlines.shape, data=coastlines)
# f.close()


############### Masked pattern correlation
# val_cut = "_cut"
# f_a = h5py.File(
#    f"{cfg.val_dir}/{cfg.save_part}/validation_{cfg.resume_iter}_assimilation_{cfg.mask_argo}_{cfg.eval_im_year}{val_cut}.hdf5",
#    "r",
# )
# f_o = h5py.File(
#    f"{cfg.val_dir}{cfg.save_part}/validation_{cfg.resume_iter}_observations_{cfg.mask_argo}_{cfg.eval_im_year}{val_cut}.hdf5",
#    "r",
# )
#
# del_t = 12
# image_o = np.nanmean(np.array(f_o.get("image")), axis=1)
# image_a = np.nanmean(np.array(f_a.get("image")), axis=1)
#
# image_o = evalu.running_mean_std(image_o, mode="mean", del_t=del_t)
# image_a = evalu.running_mean_std(image_a, mode="mean", del_t=del_t)
#
# corr = []
# for i in range(image_o.shape[0]):
#    o = image_o[i, :, :].flatten()
#    a = image_a[i, :, :].flatten()
#    corr.append(pearsonr(o, a)[0])
#
#
# length = len(corr)
# end = 2021 - (del_t // 12) // 2
# ticks = np.arange(0, length, 12 * 5)
# labels = np.arange(1958, end, 5)
#
#
## calculate running mean, if necessary
# plt.figure(figsize=(10, 6))
# plt.plot(corr, label="Pattern Correlation: Masked Assimilation - Observations")
# plt.grid()
# plt.legend()
# plt.ylim(0, 1)
# plt.xticks(ticks=ticks, labels=labels)
# plt.title("OHC Masked Pattern Correlation: Annual means")
# plt.xlabel("Time in years")
# plt.ylabel(f"Pattern Correlation as ACC")
# plt.savefig(
#    f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/masked_pattern_corr_{cfg.eval_im_year}.pdf"
# )
# plt.show()


############## EN4 reanalysis plotting

# file = f"{cfg.im_dir}En4_reanalysis_1950_2020_NA"
# ds = xr.load_dataset(f"{file}.nc", decode_times=False)
# time = ds.time
# ds["time"] = netCDF4.num2date(time[:], time.units)
# ds = ds.sel(time=slice("1958-01", "2020-10"))
#
# tos = ds.thetao.values
#
# f = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo.hdf5",
#    "r",
# )
# tos_mean = f.get("sst_mean_newgrid")
# for i in range(len(tos)):
#    tos[i] = tos[i] - tos_mean[i % 12]
#
### adjust shape of variables to fit quadratic input
# n = tos.shape
# rest = np.zeros((n[0], n[1], 128 - n[2], n[3]))
# tos = np.concatenate((tos, rest), axis=2)
# n = tos.shape
# rest2 = np.zeros((n[0], n[1], n[2], 128 - n[3]))
# tos = np.concatenate((tos, rest2), axis=3)
#
# ohc_en4 = evalu.heat_content_single(tos[:, :20, :, :])
#
## f_en4 = h5py.File(f"{file}.h5py", "r")
## ohc_en4 = np.array(f_en4.get("ohc"))
#
# plt.imshow(np.nanmean(ohc_en4[:120, :, :], axis=0), cmap="coolwarm")
# plt.show()
#
# plt.plot(np.nansum(ohc_en4, axis=(1, 2)))
# plt.show()
#
# f = h5py.File(f"{file}.hdf5", "w")
# f.create_dataset(name="ohc", shape=ohc_en4.shape, data=ohc_en4)
# f.close()
#
# ohc_newgrid = np.array(evalu.area_cutting_single(ohc_en4))
# plt.plot(np.nansum(ohc_newgrid, axis=(1, 2)))
# plt.show()
#
# f_2 = h5py.File(f"{file}_cut.hdf5", "w")
# f_2.create_dataset(name="ohc", shape=ohc_newgrid.shape, data=ohc_newgrid)
# f_2.close()


################ Valerror plotting

# error_overall = np.zeros(
#    shape=(cfg.in_channels, cfg.resume_iter // cfg.save_model_interval - 1)
# )
# for part in np.arange(cfg.combine_start, cfg.combine_start + cfg.in_channels):
#    file = f"{cfg.val_dir}part_{part}/val_errors.hdf5"
#    f = h5py.File(file, "r")
#    rsmes = np.array(f.get("rsmes")).flatten()
#    error_overall[part - cfg.combine_start, :] = rsmes
#
# error = np.mean(error_overall, axis=0)
# xx = np.arange(cfg.save_model_interval, cfg.resume_iter, cfg.save_model_interval)
# print(xx.shape)
#
# plt.figure(figsize=(10, 6))
# plt.plot(xx, error)
# plt.grid()
# plt.show()

############## IAP reanalysis plotting

# file = f"{cfg.im_dir}IAP/IAP_2000m_1958_2021"
# ds = xr.load_dataset(f"{file}.nc", decode_times=False)
# depth = ds.depth_std.values
# print(depth)
# tos = ds.temp.values
# tos = np.transpose(tos, (0, 3, 1, 2))
# tos = evalu.area_cutting_single(tos)
# print(tos.shape)
# ohc_iap = evalu.heat_content_single(tos[:, :27, :, :], depths=depth[:27])
#
# plt.imshow(np.nanmean(ohc_iap[:120, :, :], axis=0), cmap="coolwarm")
# plt.show()
#
# plt.plot(np.nansum(ohc_iap, axis=(1, 2)))
# plt.show()
#
# f = h5py.File(f"{file}.hdf5", "w")
# f.create_dataset(name="ohc", shape=ohc_iap.shape, data=ohc_iap)
# f.close()
#
# ohc_newgrid = evalu.area_cutting_single(ohc_iap)
# plt.plot(np.nansum(ohc_newgrid, axis=(1, 2)))
# plt.show()
#
# f = h5py.File(f"{file}_cut.hdf5", "w")
# f.create_dataset(name="ohc", shape=ohc_newgrid.shape, data=ohc_newgrid)
# f.close()

############# Create cut baseline_climatology
# ds = xr.load_dataset(f"{cfg.im_dir}Image_r3_14_newgrid.nc", decode_times=False)
# time_var = ds.time
# ds["time"] = netCDF4.num2date(time_var[:], time_var.units)
# ds = ds.sel(time=slice("1958-01", "2020-10"))
# ds = ds.groupby("time.month").mean("time")
#
# tos = np.array(ds.thetao.values)
# print(tos.shape)
#
#### adjust shape of variables to fit quadratic input
# n = tos.shape
# print(n)
# rest = np.zeros((n[0], n[1], 128 - n[2], n[3]))
# tos_new = np.concatenate((tos, rest), axis=2)
# n = tos_new.shape
# rest2 = np.zeros((n[0], n[1], n[2], 128 - n[3]))
# tos_new = np.concatenate((tos_new, rest2), axis=3)
# print(tos_new.shape, tos.shape)
#
#
# plt.plot(np.nanmean(tos_new, axis=(3, 2, 1)))
# plt.show()
#
# tos_cut = evalu.area_cutting_single(tos_new)
#
# f2 = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo.hdf5",
#    "w",
# )
# f2.create_dataset(name="sst_mean_newgrid", shape=tos.shape, data=tos)
# f2.create_dataset(name="sst_mean", shape=tos_new.shape, data=tos_new)
# f2.close()
#
# f = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo_cut.hdf5",
#    "w",
# )
# f.create_dataset(name="sst_mean", shape=tos_cut.shape, data=tos_cut)
# f.close()

########## plotting comparison image
#
#
# f = h5py.File(
#    f"{cfg.val_dir}part_19/heatcontent_550000_assimilation_anhang_{cfg.eval_im_year}.hdf5",
#    "r",
# )
# f_a = h5py.File(
#    f"{cfg.val_dir}part_18/heatcontent_585000_observations_anhang_{cfg.eval_im_year}.hdf5",
#    "r",
# )
#
# f_cm = h5py.File(f"{cfg.mask_dir}Kontinent_newgrid.hdf5")
# f_spg = h5py.File(f"{cfg.mask_dir}SPG_Maske.hdf5")
# spg = f_spg.get("SPG")
# continent_mask = np.array(f_cm.get("continent_mask"))
# coastlines = np.array(f_cm.get("coastlines"))
#
#
# time_1 = 732
# time_2 = 744
#
# hc_assi = np.nanmean(
#    np.nan_to_num(np.array(f.get("hc_net"))[time_1:time_2, :, :], nan=1),
#    axis=0,
# )
# hc_argo = np.nanmean(
#    np.nan_to_num(np.array(f_a.get("hc_net"))[time_1:time_2, :, :], nan=1),
#    axis=0,
# )
# hc_gt = np.nanmean(
#    np.nan_to_num(np.array(f.get("hc_gt"))[time_1:time_2, :, :], nan=1),
#    axis=0,
# )
#
#
# cmap_1 = plt.cm.get_cmap("coolwarm").copy()
# cmap_1.set_bad(color="darkgrey")
# cmap_2 = plt.cm.get_cmap("bwr").copy()
# cmap_2.set_bad(color="black")
#
# fig = plt.figure(figsize=(15, 8), constrained_layout=True)
# fig.suptitle("North Atlantic Heat Content Comparison")
# plt.subplot(1, 3, 1)
# plt.title(f"Assimilation Heat Content")
# plt.imshow(
#    hc_gt * coastlines * spg,
#    cmap=cmap_1,
#    vmin=-3e9,
#    vmax=3e9,
#    aspect="auto",
#    interpolation=None,
# )
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.subplot(1, 3, 2)
# plt.title("Network OHC (Argo Mask Training)")
# plt.imshow(
#    hc_argo * spg * coastlines,
#    cmap=cmap_1,
#    vmin=-3e9,
#    vmax=3e9,
#    aspect="auto",
#    interpolation=None,
# )
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.subplot(1, 3, 3)
# plt.title("Network OHC (Full Mask Training)")
# plt.imshow(
#    hc_assi * spg * coastlines,
#    cmap=cmap_1,
#    vmin=-3e9,
#    vmax=3e9,
#    aspect="auto",
#    interpolation=None,
# )
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(label="Heat Content in J")
# plt.savefig(
#    f"../Asi_maskiert/pdfs/validation/part_19/heat_content_2019_550000_comparison.pdf",
#    dpi=fig.dpi,
# )
# plt.show()

########## mask comparison plotting
######### non anomaly plots, north west corner

# time = 744
#
# fc = h5py.File(
#     f"{cfg.val_dir}part_19/validation_550000_assimilation_anhang_{cfg.eval_im_year}.hdf5",
#     "r",
# )
# fc_2 = h5py.File(
#     f"{cfg.val_dir}part_18/validation_585000_assimilation_anhang_{cfg.eval_im_year}.hdf5",
#     "r",
# )
# fc_o = h5py.File(
#     f"{cfg.val_dir}part_19/validation_550000_observations_anhang_{cfg.eval_im_year}.hdf5",
#     "r",
# )
# fc_o2 = h5py.File(
#     f"{cfg.val_dir}part_18/validation_585000_observations_anhang_{cfg.eval_im_year}.hdf5",
#     "r",
# )
# fa = h5py.File(
#     f"{cfg.val_dir}part_19/heatcontent_550000_assimilation_anhang_{cfg.eval_im_year}.hdf5",
#     "r",
# )
# f_en4 = h5py.File(f"{cfg.im_dir}EN4_1950_2021_NA_own.hdf5")
# en4 = np.nan_to_num(np.array(f_en4.get("ohc")), nan=1)
#
# print(en4.shape)
# time_1 = 0
# time_2 = 120
#
# f_cm = h5py.File(f"{cfg.mask_dir}Kontinent_newgrid.hdf5")
# f_spg = h5py.File(f"{cfg.mask_dir}SPG_Maske.hdf5")
# spg = f_spg.get("SPG")
# continent_mask = np.array(f_cm.get("continent_mask"))
# coastlines = np.array(f_cm.get("coastlines"))
#
# #### anomaly heatcontent
# hc_a = np.array(fa.get("hc_net"))
# hc_gt = np.array(fa.get("hc_gt"))
#
# #############try to construct nw corner mask
# nw_corner = np.zeros(shape=(128, 128))
# nw_corner[55:70, 50:70] = 1
# nw_mask = np.where(nw_corner == 1, np.nan, 1)
#
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(nw_mask)
# plt.subplot(1, 2, 2)
# plt.imshow(spg * coastlines * nw_corner)
# plt.show()
#
# fnw = h5py.File(f"{cfg.mask_dir}nw_mask.hdf5", "w")
# fnw.create_dataset(name="nw_mask", shape=nw_mask.shape, data=nw_mask)
# fnw.close()
#
#
# #### full reconstructions part 19
# image_o = np.array(fc_o.get("image"))
# mask = np.nan_to_num(np.array(fc.get("mask")), nan=0)[:, 0, :, :]
# image_o = np.array(fc_o.get("image"))
# output = np.array(fc.get("output"))
# gt = np.array(fc.get("gt"))
# output_o = np.array(fc_o.get("output"))
#
# #### argo reconstructions part 18
# output_argo = np.array(fc_2.get("output"))
# output_argo_o = np.array(fc_o2.get("output"))

# hc_a_full = evalu.heat_content_single(output, depths=True, anomalies=True, month=13)
# hc_gt_full = evalu.heat_content_single(gt, depths=True, anomalies=True, month=13)
# hc_o_full = evalu.heat_content_single(output_o, depths=True, anomalies=True, month=13)
# hc_a_full_argo = evalu.heat_content_single(
#    output_argo, depths=True, anomalies=True, month=13
# )
# hc_o_full_argo = evalu.heat_content_single(
#    output_argo_o, depths=True, anomalies=True, month=13
# )
#
# f_full = h5py.File(f"{cfg.val_dir}part_19/hc_550000_full.hdf5", "w")
# f_full.create_dataset(name="gt", shape=hc_gt_full.shape, data=hc_gt_full)
# f_full.create_dataset(name="output", shape=hc_a_full.shape, data=hc_a_full)
# f_full.create_dataset(name="output_o", shape=hc_o_full.shape, data=hc_o_full)
# f_full.create_dataset(
#    name="output_argo", shape=hc_a_full_argo.shape, data=hc_a_full_argo
# )
# f_full.create_dataset(
#    name="output_argo_o", shape=hc_o_full_argo.shape, data=hc_o_full_argo
# )
# f_full.close()

# f_full = h5py.File(f"{cfg.val_dir}part_19/hc_550000_full.hdf5", "r")
# hc_a_full = np.array(f_full.get("output"))
# hc_gt_full = np.array(f_full.get("gt"))
# hc_o_full = np.array(f_full.get("output_o"))
# hc_a_full_argo = np.array(f_full.get("output_argo"))
# hc_o_full_argo = np.array(f_full.get("output_argo_o"))
# f_full.close()
#
# print(image_o.shape)
# image_1 = image_o[0, 0, :, :]
# image_2 = image_o[time, 0, :, :]
# hc_a = np.nanmean(hc_a_full[time_1:time_2, :, :], axis=0)
# hc_gt = np.nanmean(hc_gt_full[time_1:time_2, :, :], axis=0)
#
#
# cmap_1 = plt.cm.get_cmap("seismic").copy()
# cmap_1.set_bad(color="darkgrey")
# cmap_2 = plt.cm.get_cmap("bwr").copy()
# cmap_2.set_bad(color="black")
# lines = spg * coastlines * nw_mask
# lines = np.nan_to_num(lines, nan=3)
# line = np.where(lines == 3)
# mask_none = np.where(mask == 0, np.nan, 1)
#
#
# fig = plt.figure(figsize=(12, 12), constrained_layout=True)
# fig.suptitle("Comparison: Observational Density")
# plt.subplot(2, 2, 1)
# plt.title("Neural Network OHC 1960s")
# im_1 = plt.imshow(
#     np.nanmean(hc_a_full[0:60, :, :], axis=0) * spg * coastlines * continent_mask,
#     cmap=cmap_1,
#     vmin=1e10,
#     vmax=3e10,
#     aspect="auto",
#     interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.subplot(2, 2, 2)
# plt.title("Assimilation OHC 1960s")
# plt.imshow(
#     np.nanmean(hc_gt_full[0:60, :, :], axis=0) * spg * coastlines * continent_mask,
#     cmap=cmap_1,
#     vmin=1e10,
#     vmax=3e10,
#     aspect="auto",
#     interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.subplot(2, 2, 3)
# plt.title("Neural Network OHC 2010s")
# plt.imshow(
#     np.nanmean(hc_a_full[680:740, :, :], axis=0) * spg * coastlines * continent_mask,
#     cmap=cmap_1,
#     vmin=1e10,
#     vmax=3e10,
#     aspect="auto",
#     interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.subplot(2, 2, 4)
# plt.title("Assimilation OHC 2010s")
# plt.imshow(
#     np.nanmean(hc_gt_full[680:740, :, :], axis=0) * spg * coastlines * continent_mask,
#     cmap=cmap_1,
#     vmin=1e10,
#     vmax=3e10,
#     aspect="auto",
#     interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.savefig(
#     f"../Asi_maskiert/pdfs/validation/part_19/nw_corner.pdf",
#     dpi=fig.dpi,
# )
# plt.show()


###### full values at 2020

# fig = plt.figure(figsize=(12, 11), constrained_layout=True)
# fig.suptitle("Comparison: Observational Density")
# plt.subplot(2, 2, 1)
# plt.title("EN4 OHC at Points of Observations")
# im_1 = plt.imshow(
#    en4[744] * mask_none[744] * continent_mask,
#    cmap=cmap_1,
#    vmin=0e10,
#    vmax=6e10,
#    aspect="auto",
#    interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.subplot(2, 2, 2)
# plt.title("Assimilation Heat Content")
# plt.imshow(
#    hc_gt_full[744] * spg * coastlines * continent_mask,
#    cmap=cmap_1,
#    vmin=0e10,
#    vmax=6e10,
#    aspect="auto",
#    interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.subplot(2, 2, 3)
# plt.title("Neural Network Reconstruction OHC")
# plt.imshow(
#    hc_a_full_argo[744] * spg * coastlines * continent_mask,
#    cmap=cmap_1,
#    vmin=0e10,
#    vmax=6e10,
#    aspect="auto",
#    interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.subplot(2, 2, 4)
# plt.title("Neural Network Direct Reconstruction OHC")
# plt.imshow(
#    hc_o_full_argo[744] * spg * coastlines * continent_mask,
#    cmap=cmap_1,
#    vmin=0e10,
#    vmax=6e10,
#    aspect="auto",
#    interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.savefig(
#    f"../Asi_maskiert/pdfs/validation/part_19/744_all.pdf",
#    dpi=fig.dpi,
# )
# plt.show()
#
#
# fig = plt.figure(figsize=(12, 6), constrained_layout=True)
# fig.suptitle("Comparison: Observational Density")
# plt.subplot(1, 2, 1)
# plt.title("OHC at Observations Points: January 1958")
# plt.imshow(
#    en4[0] * mask_none[0] * spg * coastlines * continent_mask,
#    cmap=cmap_1,
#    vmin=0e10,
#    vmax=6e10,
#    aspect="auto",
#    interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.subplot(1, 2, 2)
# plt.title("OHC at Observation Points: January 2020")
# plt.imshow(
#    en4[744] * mask_none[744] * spg * coastlines * continent_mask,
#    cmap=cmap_1,
#    vmin=0e10,
#    vmax=6e10,
#    aspect="auto",
#    interpolation=None,
# )
# plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
# plt.xlabel("Transformed Longitudes")
# plt.ylabel("Transformed Latitudes")
# plt.colorbar(mappable=im_1, label="Heat Content in J")
# plt.savefig(
#    f"../Asi_maskiert/pdfs/validation/part_19/mask_comparison.pdf",
#    dpi=fig.dpi,
# )
# plt.show()


fa = h5py.File(
    f"../Asi_maskiert/results/validation/part_19/validation_550000_assimilation_anhang_r2_full_newgrid.hdf5",
    "r",
)

output = fa.get("output")
output_new = evalu.area_cutting_single_nc(output, "output_name")


print(output_new.shape)

plt.imshow(output_new[0, 0, :, :])
plt.show()
