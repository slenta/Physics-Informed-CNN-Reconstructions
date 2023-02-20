# script to create mask of nort west corner of NAC for a 128 * 128 grid

import h5py
import config as cfg
import numpy as np
import matplotlib.pyplot as plt


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
nw_corner[55:70, 50:70] = 1
nw_mask = np.where(nw_corner == 1, np.nan, 1)

plt.figure(figsize=(16, 16))
plt.subplot(2, 2, 1)
plt.imshow(nw_mask)
plt.subplot(2, 2, 2)
plt.imshow(spg * coastlines * nw_corner)
plt.subplot(2, 2, 3)
plt.imshow(hc_gt_full[50, :, :])
plt.subplot(2, 2, 4)
plt.imshow(hc_gt_full * nw_corner * spg * coastlines)
plt.show()

# fnw = h5py.File(f"{cfg.mask_dir}nw_mask_60705070.hdf5", "w")
# fnw.create_dataset(name="nw_mask", shape=nw_mask.shape, data=nw_mask)
# fnw.close()

#### record book:
# 1st nw corner mask: 55, 70, 50, 70
# 2nd nw corner mask: 60, 70, 50, 70
# 3rd nw corner mask: 55, 70, 55, 70
