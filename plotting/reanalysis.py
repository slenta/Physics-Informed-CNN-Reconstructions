# script to plot iap, en4 and assimilation

import config as cfg
import matplotlib.pyplot as plt
import evaluation_og as evalu
import numpy as np
import h5py
import xarray as xr
import netCDF4

cfg.set_train_args()

############## IAP reanalysis plotting
#
# file = f"{cfg.im_dir}IAP/IAP_2000m_1958_2021_NA_newgrid_newdepth"
# ds = xr.load_dataset(f"{file}.nc", decode_times=False)
# depth = ds.depth_std.values
# print(depth)
# tos = ds.temp.values
# tos = np.transpose(tos, (0, 3, 1, 2))
#
# f = h5py.File(
#    "../Asi_maskiert/original_image/baseline_climatologyargo.hdf5",
#    "r",
# )
#
## anomalies
# ds_monthly = ds.groupby("time.month").mean("time")
# tos_mean = ds_monthly.thetao.values
#
# for i in range(len(tos)):
#    tos[i] = tos[i] - tos_mean[i % 12]
#
#
# tos_cut = evalu.area_cutting_single(tos)
# print(tos.shape)
# ohc_iap = evalu.heat_content_single(tos[:, :20, :, :], depths=depth[:27])
# ohc_iap_cut = evalu.heat_content_single(tos_cut[:, :20, :, :], depths=depth[:27])
#
# plt.imshow(np.nanmean(ohc_iap[:120, :, :], axis=0), cmap="coolwarm")
# plt.savefig("../Asi_maskiert/pdfs/misc/iap_map.pdf")
# plt.show()
#
# ohc_time = np.nansum(ohc_iap, axis=(1, 2))
# print(ohc_time.shape)
# x = range(len(ohc_time))
# plt.figure(figsize=(10, 6))
# plt.plot(x, ohc_time)
# plt.grid()
# plt.savefig("../Asi_maskiert/pdfs/misc/iap_timeseries.pdf")
# plt.show()
#
# f = h5py.File(f"{file}_own.hdf5", "w")
# f.create_dataset(name="ohc", shape=ohc_iap.shape, data=ohc_iap)
# f.close()
#
# print(ohc_iap.shape)
#
# f = h5py.File(f"{file}_own_cut.hdf5", "w")
# f.create_dataset(name="ohc", shape=ohc_iap_cut.shape, data=ohc_iap_cut)
# f.close()

############# EN4 reanalysis plotting

file = f"{cfg.im_dir}EN4_1950_2021_NA"
ds = xr.load_dataset(f"{file}.nc", decode_times=False)
time = ds.time
ds["time"] = netCDF4.num2date(time[:], time.units)
depth = ds.depth.values[:20]
ds = ds.sel(time=slice("1958-01", "2021-12"))

tos = np.array(ds.thetao.values)[:, :, :, :]
tos_a = np.zeros(tos.shape)
print(tos.shape)

ds_monthly = ds.groupby("time.month").mean("time")
tos_mean = ds_monthly.thetao.values
for i in range(len(tos)):
    tos_a[i] = tos[i] - tos_mean[i % 12]

### adjust shape of variables to fit quadratic input
n = tos.shape
rest = np.zeros((n[0], n[1], 128 - n[2], n[3]))
tos = np.concatenate((tos, rest), axis=2)
tos_a = np.concatenate((tos_a, rest), axis=2)
n = tos.shape
rest2 = np.zeros((n[0], n[1], n[2], 128 - n[3]))
tos = np.concatenate((tos, rest2), axis=3)
tos_a = np.concatenate((tos_a, rest2), axis=3)
print(tos_a.shape)
ohc_en4 = evalu.heat_content_single(tos[:, :20, :, :], depth)
print(ohc_en4.shape)
ohc_en4_a = evalu.heat_content_single(tos_a[:, :20, :, :], depth)

plt.imshow(np.nanmean(ohc_en4[:120, :, :], axis=0), cmap="coolwarm")
plt.colorbar()
plt.show()

plt.plot(np.nansum(ohc_en4, axis=(1, 2)))
plt.show()

f = h5py.File(f"{file}_own.hdf5", "w")
f.create_dataset(name="ohc", shape=ohc_en4.shape, data=ohc_en4)
f.close()

f = h5py.File(f"{file}_own_anomalies.hdf5", "w")
f.create_dataset(name="ohc", shape=ohc_en4_a.shape, data=ohc_en4_a)
f.close()

ohc_newgrid = np.array(evalu.area_cutting_single(ohc_en4))
plt.plot(np.nansum(ohc_newgrid, axis=(1, 2)))
plt.show()

f_2 = h5py.File(f"{file}_own_cut.hdf5", "w")
f_2.create_dataset(name="ohc", shape=ohc_newgrid.shape, data=ohc_newgrid)
f_2.close()


############## plot all OHC estimates
# cfg.set_train_args()
#
# file_iap = f"{cfg.im_dir}IAP/IAP_2000m_1958_2021_NA_newgrid_newdepth"
# file_en4 = f"{cfg.im_dir}En4_1950_2021_NA_newgrid"
# f_iap = h5py.File(f"{file_iap}_cut.hdf5", "r")
# f_en4 = h5py.File(f"{file_en4}_cut.hdf5", "r")
# f = h5py.File(
#    f"{cfg.val_dir}part_19/timeseries_550000_assimilation_anhang_r2_full_newgrid_cut.hdf5",
#    "r",
# )
# fo = h5py.File(
#    f"{cfg.val_dir}part_19/timeseries_550000_observations_anhang_r2_full_newgrid_cut.hdf5",
#    "r",
# )
#
# del_t = 12
#
#
# hc_a = np.array(f.get("net_ts"))
# hc_o = np.array(fo.get("net_ts"))
# hc_all_a, hc_all_o = evalu.hc_ml_ensemble(
#    members=15, part="part_19", iteration=550000, length=764
# )
#
# del_a = np.std(hc_all_a, axis=0)
# del_o = np.std(hc_all_a, axis=0)
#
# en4 = np.nansum(np.array(f_en4.get("ohc")), axis=(1, 2))
# iap = np.nansum(np.array(f_iap.get("ohc")), axis=(1, 2))
# assim_mean, std_assim, assim_all = evalu.hc_ensemble_mean_std(
#    path=cfg.im_dir, name="Image_r", members=16, length=767
# )
#
# hc_a = evalu.running_mean_std(hc_a, mode="mean", del_t=del_t)
# iap = evalu.running_mean_std(iap, mode="mean", del_t=del_t)
# hc_o = evalu.running_mean_std(hc_o, mode="mean", del_t=del_t)
# en4 = evalu.running_mean_std(en4, mode="mean", del_t=del_t)
# assim_mean = evalu.running_mean_std(assim_mean, mode="mean", del_t=del_t)
# std_assim = evalu.running_mean_std(std_assim, mode="mean", del_t=del_t)
# del_a = evalu.running_mean_std(del_a, mode="mean", del_t=del_t)
# del_o = evalu.running_mean_std(del_o, mode="mean", del_t=del_t)
#
# start = 1958 + (del_t // 12) // 2
# end = 2022 - (del_t // 12) // 2
# length = len(assim_mean)
# ticks = np.arange(0, length, 12 * 5)
# labels = np.arange(start, end, 5)
#
# plt.figure(figsize=(12, 8))
# plt.title("Summary: Different SPG OHC Estimates")
# plt.plot(en4, label="EN4 Objective Analysis", color="purple")
# plt.plot(iap, label="IAP Assimilation", color="orange")
# plt.plot(assim_mean, label="EnKF Assimilation Ensemble Mean", color="darkred")
# plt.fill_between(
#    range(len(assim_mean)),
#    assim_mean + std_assim,
#    assim_mean - std_assim,
#    label="Ensemble Spread Assimilation",
#    color="lightcoral",
# )
# plt.plot(hc_o, label="Directly Reconstructed Network OHC", color="darkgreen")
# plt.fill_between(
#    range(len(hc_o)),
#    hc_o + del_o,
#    hc_o - del_o,
#    label="Ensemble Spread Direct Reconstruction",
#    color="lightgreen",
# )
# plt.plot(hc_a, label="Indirectly Reconstructed Network OHC", color="royalblue")
# plt.fill_between(
#    range(len(hc_a)),
#    hc_a + del_a,
#    hc_a - del_a,
#    label="Ensemble Spread Indirect Reconstruction",
#    color="lightsteelblue",
# )
# plt.grid()
# plt.legend()
# plt.xticks(ticks=ticks, labels=labels)
# plt.xlabel("Time in years")
# plt.ylabel("OHC in J")
# plt.savefig(f"../Asi_maskiert/pdfs/summary_all_estimates.pdf")
# plt.show()
