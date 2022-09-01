#trying to find the fault in the observations reconstruction
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

val_path = '../Asi_maskiert/results/validation/'
part = 'part_1'
val_assim = 'validation_200000_assimilation_full.hdf5'
val_obs = 'validation_200000_observations_full.hdf5'

pdf_path = '../Asi_maskiert/pdfs/'
iteration = '200000'
argo = 'argo'


prepro = '../Asi_maskiert/original_masks/Maske_1958_2021_newgrid'
addons = '_depth_anomalies_full_20'
mixed = '../Asi_maskiert/original_image/Mixed_r16_newgrid'

mixed = f'{mixed}{addons}.hdf5'
prepro_obs = f'{prepro}{addons}_observations.hdf5'
prepro = f'{prepro}{addons}_1.hdf5'
assim = f'{val_path}{part}/{val_assim}'
obs = f'{val_path}{part}/{val_obs}'

f_assim = h5.File(assim, 'r')
f_obs = h5.File(obs, 'r')
f_prepro = h5.File(prepro, 'r')
f_prepro_obs = h5.File(prepro_obs, 'r')
f_mixed = h5.File(mixed, 'r')

mixed = f_mixed.get('tos_sym')[600, 10, :, :]
prepro_mask = f_prepro.get('tos_sym')[600, 10, :, :]
prepro_obs = f_prepro_obs.get('tos_sym')[600, 10, :, :]

output_a = f_assim.get('output')[600, 10, :, :]
gt_a = f_assim.get('gt')[600, 10, :, :]
output_comp_a = f_assim.get('output_comp')[600, 10, :, :]
image_a = f_assim.get('image')[600, 10, :, :]
mask_a = f_assim.get('mask')[600, 10, :, :]

output_o = f_obs.get('output')[600, 10, :, :]
gt_o = f_obs.get('gt')[600, 10, :, :]
output_comp_o = f_obs.get('output_comp')[600, 10, :, :]
image_o = f_obs.get('image')[600, 10, :, :]
mask_o = f_obs.get('mask')[600, 10, :, :]


fig = plt.figure(figsize=(12, 4), constrained_layout=True)
fig.suptitle('Mask Comparison')
plt.subplot(1, 3, 1)
plt.title('Image Observation')
current_cmap = plt.cm.jet
current_cmap.set_bad(color='gray')
im1 = plt.imshow(image_o, cmap=current_cmap, vmin=-4, vmax=3, aspect='auto', interpolation=None)
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 2)
plt.title('Assimilation Image')
im2 = plt.imshow(image_a, cmap = 'jet', vmin=-4, vmax=3, aspect = 'auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 3)
plt.title('Observation Mask')
im3 = plt.imshow(mask_o, cmap='jet', vmin=-3, vmax=3, aspect='auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
plt.colorbar(label='Temperature in °C')
plt.savefig(f'{pdf_path}results/Network_Input_Maske_{argo}_{iteration}_{addons}.pdf')
plt.show()

fig = plt.figure(figsize=(12, 4), constrained_layout=True)
fig.suptitle('Anomaly North Atlantic SSTs')
plt.subplot(1, 3, 1)
plt.title('Observations Output')
current_cmap = plt.cm.jet
current_cmap.set_bad(color='gray')
im1 = plt.imshow(output_o, cmap=current_cmap, vmin=-4, vmax=3, aspect='auto', interpolation=None)
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 2)
plt.title('Assimilation Output')
im2 = plt.imshow(output_a, cmap = 'jet', vmin=-4, vmax=3, aspect = 'auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 3)
plt.title('Ground Truth')
im3 = plt.imshow(gt_o, cmap='jet', vmin=-3, vmax=3, aspect='auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
plt.colorbar(label='Temperature in °C')
plt.savefig(f'{pdf_path}results/Network_results_Maske_{argo}_{iteration}_{addons}.pdf')
plt.show()



#fig = plt.figure(figsize=(12, 4), constrained_layout=True)
#fig.suptitle('Anomaly North Atlantic SSTs')
#plt.subplot(1, 3, 1)
#plt.title('Masked Image')
#current_cmap = plt.cm.jet
#current_cmap.set_bad(color='gray')
#im1 = plt.imshow(image_o, cmap=current_cmap, vmin=-3, vmax=3, aspect='auto', interpolation=None)
#plt.xlabel('Transformed Longitudes')
#plt.ylabel('Transformed Latitudes')
##plt.colorbar(label='Temperature in °C')
#plt.subplot(1, 3, 2)
#plt.title('NN Output')
#im2 = plt.imshow(output_o, cmap = 'jet', vmin=-3, vmax=3, aspect = 'auto')
#plt.xlabel('Transformed Longitudes')
#plt.ylabel('Transformed Latitudes')
##plt.colorbar(label='Temperature in °C')
#plt.subplot(1, 3, 3)
#plt.title('Original Assimilation Image')
#im3 = plt.imshow(gt_o, cmap='jet', vmin=-3, vmax=3, aspect='auto')
#plt.xlabel('Transformed Longitudes')
#plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
#plt.show()