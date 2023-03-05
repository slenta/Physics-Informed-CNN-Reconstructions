# script to create mask of nort west corner of NAC for a 128 * 128 grid

from scipy.stats import pearsonr
import h5py
import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import evaluation_og as evalu

cfg.set_train_args()

### negative: integrate false assimilation into nn output
fa = h5py.File(
    f"{cfg.val_dir}part_19/heatcontent_550000_assimilation_anhang_{cfg.eval_im_year}.hdf5",
    "r",
)

f_cm = h5py.File(f"{cfg.mask_dir}Kontinent_newgrid.hdf5")
f_spg = h5py.File(f"{cfg.mask_dir}SPG_Maske.hdf5")
spg = f_spg.get("SPG")
continent_mask = np.array(f_cm.get("continent_mask"))
coastlines = np.array(f_cm.get("coastlines"))

#### full heatcontent
f_full = h5py.File(f"{cfg.val_dir}part_19/hc_550000_full.hdf5", "r")
hc_a_full = np.array(f_full.get("output"))
hc_gt_full = np.array(f_full.get("gt"))
f_full.close()

#### anomaly heatcontent
hc_a = np.array(fa.get("hc_net"))
hc_gt = np.array(fa.get("hc_gt"))

#############try to construct nw corner mask
nw_corner = np.zeros(shape=(128, 128))
nw_corner[60:75, 53:70] = 1
nw_mask_2 = np.where(nw_corner == 1, np.nan, 1)
nw_name = "_5768_5568"

plt.figure(figsize=(16, 16))
plt.subplot(2, 2, 1)
plt.imshow(nw_mask_2)
plt.subplot(2, 2, 2)
plt.imshow(
    np.nanmean(hc_a[670:720, :, :], axis=0) * spg * coastlines,
    vmin=-3e9,
    vmax=3e9,
    cmap="coolwarm",
)
plt.subplot(2, 2, 3)
plt.imshow(
    np.nanmean(hc_gt[670:720, :, :], axis=0) * spg * coastlines,
    vmin=-3e9,
    vmax=3e9,
    cmap="coolwarm",
)
plt.subplot(2, 2, 4)
plt.imshow(
    np.nanmean(hc_gt[670:720, :, :], axis=0) * nw_corner * spg * coastlines,
    vmin=-3e9,
    vmax=3e9,
    cmap="coolwarm",
)
plt.show()

# fnw = h5py.File(f"{cfg.mask_dir}nw_mask{nw_name}.hdf5", "w")
# fnw.create_dataset(name="nw_mask", shape=nw_mask_2.shape, data=nw_mask_2)
# fnw.close()

#### record book:
# 1st nw corner mask: 55, 70, 50, 70
# 2nd nw corner mask: 57, 68, 55, 68
# 3rd nw corner mask: 55, 70, 55, 70


### positive: integrate nn into assimilation ensemble member
fa = h5py.File(
    f"{cfg.val_dir}part_19/heatcontent_550000_assimilation_anhang_{cfg.eval_im_year}.hdf5",
    "r",
)

f_full = h5py.File(f"{cfg.val_dir}part_19/hc_550000_full.hdf5", "r")
output = np.array(f_full.get("output"))
gt = np.array(f_full.get("gt"))
f_full.close()

out_a = fa.get("hc_net")
gt_a = fa.get("hc_gt")

nw_corner = np.zeros(shape=(128, 128))
nw_corner[30:105, 33:100] = 1
nw_mask = np.where(nw_corner == 1, 1, np.nan)
nw_inverse = np.where(nw_corner == 1, np.nan, 1)

nw_output = output * nw_mask
nw_output = np.nan_to_num(nw_output, nan=1)
nw_output_a = np.nan_to_num(out_a * nw_mask, nan=1)
nw_gt_a = np.nan_to_num(nw_inverse * gt_a, nan=1) * nw_output_a
nw_gt = np.nan_to_num(nw_inverse * gt, nan=1) * nw_output
nw_gt_ts = np.nansum(evalu.area_cutting_single(nw_gt_a), axis=(2, 1))

f_cm = h5py.File(f"{cfg.mask_dir}Kontinent_newgrid.hdf5")
f_spg = h5py.File(f"{cfg.mask_dir}SPG_Maske.hdf5")
spg = f_spg.get("SPG")
continent_mask = np.array(f_cm.get("continent_mask"))
coastlines = np.array(f_cm.get("coastlines"))
mini = 1.5e10
maxi = 2.5e10

plt.figure(figsize=(16, 16))
plt.subplot(2, 2, 1)
plt.title("NN Output")
plt.imshow(
    np.nanmean(output[0:60, :, :], axis=0) * spg * coastlines, vmin=mini, vmax=maxi
)
plt.subplot(2, 2, 2)
plt.title("Assimilation")
plt.imshow(np.nanmean(gt[0:60, :, :], axis=0) * spg * coastlines, vmin=mini, vmax=maxi)
plt.subplot(2, 2, 3)
plt.title("Assimilation + NN Nw corner")
plt.imshow(
    np.nanmean(nw_gt[0:60, :, :], axis=0) * spg * coastlines, vmin=mini, vmax=maxi
)
plt.subplot(2, 2, 4)
plt.title("NN Nw corner")
plt.imshow(
    np.nanmean(nw_output[0:60, :, :], axis=0) * spg * coastlines, vmin=mini, vmax=maxi
)
plt.show()

# save adjusted assimilation in hdf5 File
f_nw = h5py.File(f"{cfg.val_dir}part_19/hc_assi_nw_adjusted.hdf5", "w")
f_nw.create_dataset(name="hc_assi_nw", shape=nw_gt.shape, data=nw_gt)
f_nw.create_dataset(name="hc_assi_nw_ts", shape=nw_gt_ts.shape, data=nw_gt_ts)
f_nw.close()


# timeseries plotting of nw adjusted assimilation
part = "part_19"
iteration = 550000
mask_argo = "anhang"

f = h5py.File(
    f"{cfg.val_dir}{part}/timeseries_{str(iteration)}_assimilation_{mask_argo}_{cfg.eval_im_year}_cut.hdf5",
    "r",
)

ts_gt = np.array(f.get("gt_ts"))[:754]
ts_net = np.array(f.get("net_ts"))[:754]
ts_gt_annual, ts_net_annual, ts_gt_nw_annual = (
    evalu.running_mean_std(ts_gt, mode="mean", del_t=12),
    evalu.running_mean_std(ts_net, mode="mean", del_t=12),
    evalu.running_mean_std(nw_gt_ts, mode="mean", del_t=12),
)
print(ts_gt.shape, ts_net.shape, nw_gt_ts.shape)


plt.figure(figsize=(10, 6))
plt.title("NA SPG OHC")
plt.plot(
    ts_gt_annual,
    color="darkred",
    label=f"Assimilation, Correlation: {pearsonr(ts_gt, ts_net)[0]}",
)
plt.plot(ts_net_annual, color="royalblue", label="Neural Network Reconstruction")
plt.plot(
    ts_gt_nw_annual,
    color="red",
    label=f"NWC Corrected Assimilation, Correlation: {pearsonr(nw_gt_ts, ts_net)[0]}",
)
plt.grid()
plt.legend()
plt.xlabel("Time in years")
plt.ylabel("Heat Content [J]")
plt.show()


hc_all_a, hc_all_o = evalu.hc_ml_ensemble(
    members=15, part="part_19", iteration=550000, length=764
)
gt_mean, std_gt, hc_all = evalu.hc_ensemble_mean_std(
    cfg.im_dir, name="Image_r", members=16
)
del_a = evalu.running_mean_std(np.nanstd(hc_all_a, axis=0), mode="mean", del_t=12)
del_o = evalu.running_mean_std(np.nanstd(hc_all_a, axis=0), mode="mean", del_t=12)
std_gt = evalu.running_mean_std(std_gt, mode="mean", del_t=12)
del_gt = std_gt[: len(del_a)]
misfit = ts_gt_annual - ts_net_annual
del_total = (del_a + del_gt)[: len(misfit)]
misfit_nw = ts_gt_nw_annual - ts_net_annual

plt.figure(figsize=(10, 6))
plt.title("Differences NWC Correction")
plt.plot(ts_gt_annual - ts_net_annual, color="darkred", label="Misfit Assimilation, NN")
plt.plot(
    ts_gt_nw_annual - ts_net_annual,
    color="blue",
    label="Misfit NWC Corrected Assimilation, NN",
)
# plt.fill_between(
#    range(len(misfit)),
#    misfit + del_total,
#    misfit - del_total,
#    label="Combined Uncertainty",
#    color="lightsteelblue",
# )
# plt.fill_between(
#    range(len(misfit)),
#    misfit_nw + del_total,
#    misfit_nw - del_total,
#    label="Combined Uncertainty",
#    color="royalblue",
# )
plt.grid()
plt.legend()
plt.show()
