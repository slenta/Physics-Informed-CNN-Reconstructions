#trying to find the fault in the observations reconstruction
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

val_path = '../Asi_maskiert/results/validation/'
part = 'Maske_argo_mixed'
val_assim = 'validation_200000_assimilation_full.hdf5'
val_obs = 'validation_200000_observations_full.hdf5'

assim = f'{val_path}{part}/{val_assim}'
obs = f'{val_path}{part}/{val_obs}'
f_assim = h5.File(obs, 'r')

output_a = f_assim.get('output')[600, 0, :, :]
gt_a = f_assim.get('gt')[600, 0, :, :]
output_comp_a = f_assim.get('output_comp')[600, 0, :, :]
image_a = f_assim.get('image')[600, 0, :, :]
mask_a = f_assim.get('mask')[600, 0, :, :]

print(output_a.shape)


fig = plt.figure(figsize=(12, 4), constrained_layout=True)
fig.suptitle('Anomaly North Atlantic SSTs')
plt.subplot(1, 3, 1)
plt.title('Masked Image')
current_cmap = plt.cm.jet
current_cmap.set_bad(color='gray')
im1 = plt.imshow(image_a, cmap=current_cmap, vmin=-3, vmax=3, aspect='auto', interpolation=None)
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 2)
plt.title('NN Output')
im2 = plt.imshow(output_a, cmap = 'jet', vmin=-3, vmax=3, aspect = 'auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 3)
plt.title('Original Assimilation Image')
im3 = plt.imshow(gt_a, cmap='jet', vmin=-3, vmax=3, aspect='auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
plt.colorbar(label='Temperature in °C')
plt.savefig('../Asi_maskiert/pdfs/experiments/2008_obs_mixed.pdf')
plt.show()