#trying to find the fault in the observations reconstruction
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

results_path = '../Asi_maskiert/results/images/'
part = 'part_1'
pdf_path = '../Asi_maskiert/pdfs/'
iteration = '300000'
image = f'test_{iteration}.hdf5'

f = h5.File(f'{results_path}{part}/{image}', 'r')
output = f.get('output')[0, 0, :, :]
gt = f.get('gt')[0, 0, :, :]
image = f.get('image')[0, 0, :, :]

continent_mask = np.where(gt==0, np.NaN, 1)

fig = plt.figure(figsize=(12, 4), constrained_layout=True)
fig.suptitle('Mask Comparison')
plt.subplot(1, 3, 1)
plt.title('Image Observation')
current_cmap = plt.cm.jet
current_cmap.set_bad(color='gray')
im1 = plt.imshow(image * continent_mask, cmap=current_cmap, vmin=-4, vmax=3, aspect='auto', interpolation=None)
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 2)
plt.title('Network Output')
im2 = plt.imshow(output * continent_mask, cmap = 'jet', vmin=-4, vmax=3, aspect = 'auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 3, 3)
plt.title('Original Assimilation Image')
im3 = plt.imshow(gt * continent_mask, cmap='jet', vmin=-3, vmax=3, aspect='auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
plt.colorbar(label='Temperature in °C')
#plt.savefig(f'{pdf_path}results/Network_Input_Maske_{argo}_{iteration}_{addons}.pdf')
plt.show()

#fig = plt.figure(figsize=(12, 4), constrained_layout=True)
#fig.suptitle('Anomaly North Atlantic SSTs')
#plt.subplot(1, 3, 1)
#plt.title('Observations Output')
#current_cmap = plt.cm.jet
#current_cmap.set_bad(color='gray')
#im1 = plt.imshow(output_o, cmap=current_cmap, vmin=-4, vmax=3, aspect='auto', interpolation=None)
#plt.xlabel('Transformed Longitudes')
#plt.ylabel('Transformed Latitudes')
##plt.colorbar(label='Temperature in °C')
#plt.subplot(1, 3, 2)
#plt.title('Assimilation Output')
#im2 = plt.imshow(output_a, cmap = 'jet', vmin=-4, vmax=3, aspect = 'auto')
#plt.xlabel('Transformed Longitudes')
#plt.ylabel('Transformed Latitudes')
##plt.colorbar(label='Temperature in °C')
#plt.subplot(1, 3, 3)
#plt.title('Ground Truth')
#im3 = plt.imshow(gt_o, cmap='jet', vmin=-3, vmax=3, aspect='auto')
#plt.xlabel('Transformed Longitudes')
#plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
#plt.savefig(f'{pdf_path}results/Network_results_Maske_{argo}_{iteration}_{addons}.pdf')
#plt.show()
#
#
#
#plt.figure(figsize=(10, 6))
#plt.plot(T_mean_o, label='Mean Temp Observation Mask')
#plt.plot(T_mean_a, label='Mean Temp Assimilation Mask')
#plt.grid()
#plt.legend()
#plt.xticks(ticks=np.arange(0, len(T_mean_o), 5*12), labels=np.arange(1958, 2020, 5))
#plt.title('Comparison Reconstruction to Assimilation Timeseries')
#plt.xlabel('Time of observations [years]')
#plt.ylabel('Heat Content [J/m²]')
#plt.savefig(f'../Asi_maskiert/pdfs/timeseries/masked_timeseries.pdf')
#plt.show()



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