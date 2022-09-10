from cProfile import label
import h5py
from isort import file
from matplotlib.pyplot import title
import numpy as np
import pylab as plt
import torch
import xarray as xr
import config as cfg
import netCDF4
import evaluation_og as evalu



def masked_output_vis(part, iter, time, depth):

    fa = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iter}_assimilation_full.hdf5', 'r')
    fo = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iter}_observations_full.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
    
    continent_mask = np.array(fm.get('continent_mask'))[0, :, :]

    mask = np.array(fo.get('mask')[time, depth,:, :]) * continent_mask
    output_a = np.array(fa.get('output')[time, depth,:, :]) * continent_mask
    image_a = np.array(fa.get('image')[time, depth,:, :]) * continent_mask
    output_o = np.array(fo.get('output')[time, depth,:, :]) * continent_mask
    image_o = np.array(fo.get('image')[time, depth,:, :]) * continent_mask

    mask_grey = np.where(mask==0, np.NaN, mask) * continent_mask

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    fig.suptitle('Anomaly North Atlantic SSTs')
    plt.subplot(2, 2, 1)
    plt.title(f'Masked Assimilations')
    current_cmap = plt.cm.jet
    current_cmap.set_bad(color='gray')
    plt.imshow(image_a * mask_grey, cmap=current_cmap, vmin=-3, vmax=3, aspect='auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 2)
    plt.title(f'Observation Mask')
    im2 = plt.imshow(image_o * mask_grey, cmap = 'jet', vmin=-3, vmax=3, aspect = 'auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 3)
    plt.title('Masked Assimilation Reconstructions')
    plt.imshow(output_a * mask_grey, vmin=-3, vmax=3, cmap='jet', aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 4)
    plt.title('Masked Observations Reconstructions')
    plt.imshow(output_o * mask_grey, cmap='jet', vmin=-3, vmax=3, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Annomaly Correlation')
    fig.savefig(f'../Asi_maskiert/results/validation/{part}/validation_masked_{iter}_timestep_{str(time)}_depth_{str(depth)}.pdf', dpi = fig.dpi)
    plt.show()


def output_vis(part, iter, time, depth, mode):

    if mode == 'Assimilation':    
        f = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iter}_assimilation_full.hdf5', 'r')
    elif mode == 'Observations':
        f = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iter}_observations_full.hdf5', 'r')
    
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
    continent_mask = np.array(fm.get('continent_mask'))

    gt = np.array(f.get('gt')[:, depth, :, :]) * continent_mask
    mask = np.array(f.get('mask')[:, depth,:, :]) * continent_mask
    output = np.array(f.get('output')[:, depth,:, :]) * continent_mask
    image = np.array(f.get('image')[:, depth,:, :]) * continent_mask

    mask_grey = np.where(mask==0, np.NaN, mask) * continent_mask

    correlation, sig = evalu.correlation(output, gt) 


    #error = np.zeros((8, depth))
    #for i in range(8):
    #    for j in range(depth):
    #        image = f.get('image')[i, j, :, :]
    #        output = f.get('output')[i, j, :, :]
    #        mask = f.get('mask')[i, j, :, :]
    #        outputcomp = mask*image + (1 - mask)*output
    #
    #        error[i, j] = np.mean((np.array(outputcomp) - np.array(image))**2)


    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    fig.suptitle('Anomaly North Atlantic SSTs')
    plt.subplot(2, 2, 1)
    plt.title(f'Masked Image {mode}')
    current_cmap = plt.cm.jet
    current_cmap.set_bad(color='gray')
    im1 = plt.imshow(image[time, :, :] * mask_grey[time, :, :], cmap=current_cmap, vmin=-3, vmax=3, aspect='auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 2)
    plt.title(f'Reconstructed {mode} Output')
    im2 = plt.imshow(output[time, :, :], cmap = 'jet', vmin=-3, vmax=3, aspect = 'auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 3)
    plt.title('Original Assimilation Image')
    im4 = plt.imshow(gt[time, :, :], vmin=-3, vmax=3, cmap='jet', aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 4)
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color='gray')
    plt.scatter(sig[1], sig[0], c='black', s=0.7, marker='.', alpha=0.2)
    im3 = plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Annomaly Correlation')
    fig.savefig(f'../Asi_maskiert/results/validation/{part}/validation_{mode}_{iter}_timestep_{str(time)}_depth_{str(depth)}.pdf', dpi = fig.dpi)
    plt.show()


                
def timeseries_plotting(path, iteration, argo, mean='monthly'):

    f = h5py.File(f'{cfg.val_dir}{path}/timeseries_{str(iteration)}_assimilation_{argo}.hdf5', 'r')
    fo = h5py.File(f'{cfg.val_dir}{path}/timeseries_{str(iteration)}_observations_{argo}.hdf5', 'r')
 
    hc_assi = np.array(f.get('net_ts'))
    hc_gt = np.array(f.get('gt_ts'))
    hc_assi_masked = np.array(f.get('net_ts_masked'))
    hc_gt_masked = np.array(f.get('gt_ts_masked'))
    hc_obs = np.array(fo.get('net_ts'))
    hc_obs_masked = np.array(fo.get('net_ts_masked'))
    
    #define running mean timesteps
    if mean=='annual':
        del_t = 12
    elif mean=='5_year':
        del_t = 12*5
    else:
        del_t = 1

    #calculate running mean, if necessary
    if del_t != 1:
        hc_assi = evalu.running_mean_std(hc_assi, mode='mean', del_t=del_t) 
        hc_gt = evalu.running_mean_std(hc_gt, mode='mean', del_t=del_t) 
        hc_obs = evalu.running_mean_std(hc_obs, mode='mean', del_t=del_t)
        hc_assi_masked = evalu.running_mean_std(hc_assi_masked, mode='mean', del_t=del_t) 
        hc_gt_masked = evalu.running_mean_std(hc_gt_masked, mode='mean', del_t=del_t) 
        hc_obs_masked = evalu.running_mean_std(hc_obs_masked, mode='mean', del_t=del_t)

        ticks = np.arange(0, len(hc_assi), 12*5)
        labels = np.arange(1958 + (del_t/12)%2+1, 2021 - (del_t/12)%2, 5) 

    ticks = np.arange(0, len(hc_assi), 12*5)
    labels = np.arange(1958, 2021, 5) 

    plt.figure(figsize=(10, 6))

    plt.plot(hc_assi, label='Network Reconstructed Heat Content')
    plt.plot(hc_gt, label='Assimilation Heat Content')
    plt.plot(hc_obs, label='Observations reconstruction')
    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title('Comparison Reconstruction to Assimilation Timeseries')
    plt.xlabel('Time in years')
    plt.ylabel('Heat Content [J/m²]')
    plt.savefig(f'../Asi_maskiert/pdfs/timeseries/{path}/validation_timeseries_{argo}_{str(iteration)}_{mean}_mean.pdf')
    plt.show()

    plt.plot(hc_assi_masked, label='Network Reconstructed Heat Content')
    plt.plot(hc_gt_masked, label='Assimilation Heat Content')
    plt.plot(hc_obs_masked, label='Observations reconstruction')
    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title('Comparison Reconstruction to Assimilation Timeseries at Observation Points')
    plt.xlabel('Time in years')
    plt.ylabel('Heat Content [J/m²]')
    plt.savefig(f'../Asi_maskiert/pdfs/timeseries/{path}/validation_timeseries_masked_{argo}_{str(iteration)}_{mean}_mean.pdf')
    plt.show()



def std_plotting(path, iteration, argo, del_t):
   
    f = h5py.File(f'{cfg.val_dir}{path}/timeseries_{str(iteration)}_assimilation_{argo}.hdf5', 'r')
    fo = h5py.File(f'{cfg.val_dir}{path}/timeseries_{str(iteration)}_observations_{argo}.hdf5', 'r')
 
    hc_network = f.get('network_ts')
    hc_gt = f.get('gt_ts')

    hc_network = np.array(hc_network)
    hc_gt = np.array(hc_gt)
    hc_obs = np.array(fo.get('network_ts'))

    #plot running std
    hc_net_std = evalu.running_mean_std(hc_network, mode='std', del_t=del_t) 
    hc_gt_std = evalu.running_mean_std(hc_gt, mode='std', del_t=del_t) 
    hc_obs_std = evalu.running_mean_std(hc_obs, mode='std', del_t=del_t) 

    ticks = np.arange(0, len(hc_net_std), 12*5)
    labels = np.arange(1958 + (del_t/12)%2, 2021 - (del_t/12)%2, 5) 
    
    plt.figure(figsize=(10, 6))
    plt.plot(hc_net_std, label='Standard Deviation Assimilation Reconstruction')
    plt.plot(hc_gt_std, label='Standard Deviation Original Assimilation')
    plt.plot(hc_obs_std, label='Standard Deviation Observations Reconstruction')
    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title('Comparison Standard Devation of Reconstructions to Original Assimilation')
    plt.xlabel('Time in years')
    plt.ylabel(f'Standard Deviation of Heat Content ({str(del_t/12)} years)')
    plt.savefig(f'../Asi_maskiert/pdfs/timeseries/{path}/validation_std_timeseries_{argo}_{str(iteration)}_{str(del_t)}.pdf')
    plt.show()


     


cfg.set_train_args()
#masked_output_vis('part_2', '200000', time=700, depth=0)
#output_vis('part_2', '200000', time=700, depth=0, mode='Observations')
timeseries_plotting('part_2', 200000, argo='full', mean='monthly')
#std_plotting('part_1', 200000, 'full', del_t=2*12)



#vis_single(753, '../Asi_maskiert/original_image/', 'Image_3d_newgrid', 'r1011_shuffle_newgrid/short_val/Maske_1970_1985r1011_shuffle_newgrid/short_val/Maske_1970_1985Argo-era', 'image', 'image', 'North Atlantic Assimilation October 2020')
#vis_single(9, '../Asi_maskiert/original_masks/', 'Maske_2020_newgrid', 'pre-Argo-era', 'mask', 'mask', 'North Atlantic Observations October 2020')

#vis_single(1, '../Asi_maskiert/results/images/r1011_shuffle_newgrid/short_val/Maske_1970_1985/', 'test_700000', 'pre-argo-era', 'output', 'mask', 'Pre-Argo-Era Masks')