from cProfile import label
import os
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
from scipy.stats import pearsonr

def masked_output_vis(part, iter, time, depth):

    fa = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iter}_assimilation_full.hdf5', 'r')
    fo = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iter}_observations_full.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
    
    continent_mask = np.array(fm.get('continent_mask'))[0, :, :]

    mask = np.array(fa.get('mask')[time, depth,:, :]) * continent_mask
    output_a = np.array(fa.get('output')[time, depth,:, :]) * continent_mask
    image_a = np.array(fa.get('image')[time, depth,:, :]) * continent_mask
    output_o = np.array(fo.get('output')[time, depth,:, :]) * continent_mask
    image_o = np.array(fo.get('image')[time, depth,:, :]) * continent_mask

    mask_grey = np.where(mask==0, np.NaN, mask) * continent_mask
    save_dir = f'../Asi_maskiert/pdfs/validation/{part}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
    fig.savefig(f'../Asi_maskiert/pdfs/validation/{part}/validation_masked_{iter}_timestep_{str(time)}_depth_{str(depth)}.pdf', dpi = fig.dpi)
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
    T_mean_output = np.nanmean(np.nanmean(output, axis=1), axis=1)
    T_mean_gt = np.nanmean(np.nanmean(gt, axis=1), axis=1)


    #error = np.zeros((8, depth))
    #for i in range(8):
    #    for j in range(depth):
    #        image = f.get('image')[i, j, :, :]
    #        output = f.get('output')[i, j, :, :]
    #        mask = f.get('mask')[i, j, :, :]
    #        outputcomp = mask*image + (1 - mask)*output
    #
    #        error[i, j] = np.mean((np.array(outputcomp) - np.array(image))**2)


    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    fig.suptitle('Anomaly North Atlantic SSTs')
    plt.subplot(1, 3, 1)
    plt.title(f'Masked Image {mode}')
    current_cmap = plt.cm.jet
    current_cmap.set_bad(color='gray')
    im1 = plt.imshow(image[time, :, :] * mask_grey[time, :, :], cmap=current_cmap, vmin=-3, vmax=3, aspect='auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 2)
    plt.title(f'Reconstructed {mode}: Network Output')
    im2 = plt.imshow(output[time, :, :], cmap = 'jet', vmin=-3, vmax=3, aspect = 'auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 3)
    plt.title('Original Assimilation Image')
    im4 = plt.imshow(gt[time, :, :], vmin=-3, vmax=3, cmap='jet', aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    fig.savefig(f'../Asi_maskiert/pdfs/validation/{part}/validation_{mode}_{iter}_timestep_{str(time)}_depth_{str(depth)}.pdf', dpi = fig.dpi)
    plt.show()


def correlation_plotting(path, iteration, depth):

    f_a = h5py.File(f'../Asi_maskiert/results/validation/{path}/validation_{iteration}_assimilation_full.hdf5', 'r')
    f_o = h5py.File(f'../Asi_maskiert/results/validation/{path}/validation_{iteration}_observations_full.hdf5', 'r')
    
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
    continent_mask = np.array(fm.get('continent_mask'))

    gt = np.array(f_a.get('gt')[:, depth, :, :]) * continent_mask
    mask_a = np.array(f_a.get('mask')[:, depth,:, :]) * continent_mask
    output_a = np.array(f_a.get('output')[:, depth,:, :]) * continent_mask
    image_a = np.array(f_a.get('image')[:, depth,:, :]) * continent_mask
    mask_o = np.array(f_o.get('mask')[:, depth,:, :]) * continent_mask
    output_o = np.array(f_o.get('output')[:, depth,:, :]) * continent_mask
    image_o = np.array(f_o.get('image')[:, depth,:, :]) * continent_mask

    mask_grey = np.where(mask_a==0, np.NaN, mask_a) * continent_mask
    T_mean_output_a = np.nanmean(np.nanmean(output_a, axis=1), axis=1)
    T_mean_output_o = np.nanmean(np.nanmean(output_o, axis=1), axis=1)
    T_mean_gt = np.nanmean(np.nanmean(gt, axis=1), axis=1)

    correlation_argo_a, sig_argo_a = evalu.correlation(output_a[552:], gt[552:])
    correlation_preargo_a, sig_preargo_a = evalu.correlation(output_a[:552], gt[:552])
    correlation_argo_o, sig_argo_o = evalu.correlation(output_o[552:], gt[552:])
    correlation_preargo_o, sig_preargo_o = evalu.correlation(output_o[:552], gt[:552])

    acc_mean_argo_a = pearsonr(np.nanmean(np.nanmean(gt[552:], axis=1), axis=1), np.nanmean(np.nanmean(output_a[552:], axis=1), axis=1))[0]
    acc_mean_preargo_a = pearsonr(np.nanmean(np.nanmean(gt[:552], axis=1), axis=1), np.nanmean(np.nanmean(output_a[:552], axis=1), axis=1))[0]
    acc_mean_argo_o = pearsonr(np.nanmean(np.nanmean(gt[552:], axis=1), axis=1), np.nanmean(np.nanmean(output_o[552:], axis=1), axis=1))[0]
    acc_mean_preargo_o = pearsonr(np.nanmean(np.nanmean(gt[:552], axis=1), axis=1), np.nanmean(np.nanmean(output_o[:552], axis=1), axis=1))[0]

    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    fig.suptitle('Anomaly North Atlantic SSTs')
    plt.subplot(2, 2, 1)
    plt.title(f'Correlation Argo Era: Assimilation Reconstruction and Assimilation Image: {acc_mean_argo_a:.2f}')
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color='gray')
    plt.scatter(sig_argo_a[1], sig_argo_a[0], c='black', s=0.7, marker='.', alpha=0.2)
    im3 = plt.imshow(correlation_argo_a, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.subplot(2, 2, 2)
    plt.title(f'Correlation Argo Era: Observation Reconstruction and Assimilation Image: {acc_mean_argo_o:.2f}')
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color='gray')
    plt.scatter(sig_argo_o[1], sig_argo_o[0], c='black', s=0.7, marker='.', alpha=0.2)
    im3 = plt.imshow(correlation_argo_o, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.subplot(2, 2, 3)
    plt.title(f'Correlation Preargo Era: Assimilation Reconstruction and Assimilation Image: {acc_mean_preargo_a:.2f}')
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color='gray')
    plt.scatter(sig_preargo_a[1], sig_preargo_a[0], c='black', s=0.7, marker='.', alpha=0.2)
    im3 = plt.imshow(correlation_preargo_a, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.subplot(2, 2, 4)
    plt.title(f'Correlation Preargo Era: Observations Reconstruction and Assimilation Image: {acc_mean_preargo_o:.2f}')
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color='gray')
    plt.scatter(sig_preargo_o[1], sig_preargo_o[0], c='black', s=0.7, marker='.', alpha=0.2)
    im3 = plt.imshow(correlation_preargo_o, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Annomaly Correlation')
    fig.savefig(f'../Asi_maskiert/pdfs/validation/{path}/correlation_{iteration}.pdf', dpi = fig.dpi)
    plt.show()
    

def hc_plotting(path, iteration, time=600, argo='full', mean='monthly'):

    f = h5py.File(f'{cfg.val_dir}{path}/heatcontent_{str(iteration)}_assimilation_{argo}.hdf5', 'r')
    fo = h5py.File(f'{cfg.val_dir}{path}/heatcontent_{str(iteration)}_observations_{argo}.hdf5', 'r')
    fc = h5py.File(f'{cfg.val_dir}{path}/validation_{str(iteration)}_observations_{argo}_cut.hdf5', 'r')
    gt = fc.get('gt')
    continent_mask = np.where(gt==0, np.NaN, 1) 
 
    hc_assi = np.array(f.get('net_ts')) * continent_mask
    hc_gt = np.array(f.get('gt_ts')) * continent_mask
    hc_assi_masked = np.array(f.get('net_ts_masked'))
    hc_gt_masked = np.array(f.get('gt_ts_masked'))
    hc_obs = np.array(fo.get('net_ts')) * continent_mask
    hc_obs_masked = np.array(fo.get('net_ts_masked'))
    hc_obs_gt_masked = np.array(fo.get('gt_ts_masked'))

    
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
        hc_obs_gt_masked = evalu.running_mean_std(hc_obs_gt_masked, mode='mean', del_t=del_t)

        ticks = np.arange(0, len(hc_assi), 12*5)
        labels = np.arange(1958 + (del_t/12)%2+1, 2021 - (del_t/12)%2, 5) 

    ticks = np.arange(0, len(hc_assi), 12*5)
    labels = np.arange(1958, 2021, 5) 

    correlation, sig = evalu.correlation(hc_assi, hc_gt)
    correlation_argo, sig_argo = evalu.correlation(hc_assi[552:], hc_gt[552:])
    correlation_preargo, sig_preargo = evalu.correlation(hc_assi[:552], hc_gt[:552])
    acc_mean = pearsonr(np.nanmean(hc_gt, axis=(1, 2)), np.nanmean(hc_assi, axis=(1, 2))[0]

    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    fig.suptitle('Assimilation Heat Content Comparison')
    plt.subplot(1, 3, 1)
    plt.title('Assimilation Heat Content')
    current_cmap = plt.cm.jet
    current_cmap.set_bad(color='gray')
    im1 = plt.imshow(hc_gt[time, :, :], cmap=current_cmap, vmin=-5e9, vmax=5e9, aspect='auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 2)
    plt.title('Network Output Heat Content')
    im2 = plt.imshow(hc_assi[time, :, :], cmap = 'jet', vmin=-5e9, vmax=5e9, aspect = 'auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 3)
    plt.title(f'Correlation: Network Output and Assimilation Image: {acc_mean:.2f}')
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color='gray')
    plt.scatter(sig[1], sig[0], c='black', s=0.7, marker='.', alpha=0.2)
    im3 = plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Annomaly Correlation')
    plt.savefig(f'../Asi_maskiert/pdfs/validation/{path}/heat_content_correlation_{time}_{iteration}.pdf')
    plt.show()
                

def timeseries_plotting(path, iteration, argo, mean='monthly'):

    f = h5py.File(f'{cfg.val_dir}{path}/timeseries_{str(iteration)}_assimilation_{argo}.hdf5', 'r')
    fo = h5py.File(f'{cfg.val_dir}{path}/timeseries_{str(iteration)}_observations_{argo}.hdf5', 'r')

    f_a = h5py.File(f'../Asi_maskiert/results/validation/{path}/validation_{iteration}_assimilation_full_cut.hdf5', 'r')
    f_o = h5py.File(f'../Asi_maskiert/results/validation/{path}/validation_{iteration}_observations_full_cut.hdf5', 'r')
 
    hc_assi = np.array(f.get('net_ts'))
    hc_gt = np.array(f.get('gt_ts'))
    hc_assi_masked = np.array(f.get('net_ts_masked'))
    hc_gt_masked = np.array(f.get('gt_ts_masked'))
    hc_obs = np.array(fo.get('net_ts'))
    hc_obs_masked = np.array(fo.get('net_ts_masked'))
    hc_obs_gt_masked = np.array(fo.get('gt_ts_masked'))

    gt = np.array(f_a.get('gt')[:, 0, :, :]) 
    continent_mask = np.where(gt==0, np.NaN, 1)
    gt = gt * continent_mask
    output_a = np.nanmean(np.array(f_a.get('output')[:, :, :, :]), axis=1) * continent_mask
    output_o = np.nanmean(np.array(f_o.get('output')[:, :, :, :]), axis=1) * continent_mask

    T_mean_output_a = np.nanmean(output_a, axis=(1, 2))
    T_mean_output_o = np.nanmean(output_o, axis=(1, 2))
    T_mean_gt = np.nanmean(gt, axis=(1, 2))

    
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
        hc_obs_gt_masked = evalu.running_mean_std(hc_obs_gt_masked, mode='mean', del_t=del_t)

        ticks = np.arange(0, len(hc_assi), 12*5)
        labels = np.arange(1958 + (del_t/12)%2 + 1, 2021 - (del_t/12)%2 + 3, 5) 

    else:
        ticks = np.arange(0, len(hc_assi), 12*5)
        labels = np.arange(1958, 2021, 5) 

    plt.figure(figsize=(10, 6))

    plt.plot(T_mean_output_a, label='Network Reconstructed Heat Content')
    plt.plot(T_mean_gt, label='Assimilation Heat Content')
    plt.plot(T_mean_output_o, label='Observations reconstruction')
    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title('Comparison Reconstruction to Assimilation Timeseries')
    plt.xlabel('Time in years')
    plt.ylabel('Heat Content [J/m²]')
    plt.savefig(f'../Asi_maskiert/pdfs/validation/{path}/T_mean_timeseries_{argo}_{str(iteration)}_{mean}_mean.pdf')
    plt.show()

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
    plt.savefig(f'../Asi_maskiert/pdfs/validation/{path}/validation_timeseries_{argo}_{str(iteration)}_{mean}_mean.pdf')
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.plot(hc_assi_masked, label='Assimilation reconstruction')
    plt.plot(hc_gt_masked, label='Assimilation Heat Content')
    plt.plot(hc_obs_masked, label='Observations reconstruction')
    plt.plot(hc_obs_gt_masked, label='Observations Heat Content')
    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title('Comparison Reconstruction to Assimilation Timeseries at Observation Points')
    plt.xlabel('Time in years')
    plt.ylabel('Heat Content [J/m²]')
    plt.savefig(f'../Asi_maskiert/pdfs/validation/{path}/validation_timeseries_masked_{argo}_{str(iteration)}_{mean}_mean.pdf')
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

def pattern_corr_plot(part):
 
    f_o = h5py.File(f'{cfg.val_dir}{cfg.save_part}/pattern_corr_ts_{str(cfg.resume_iter)}_observations_full.hdf5', 'r')
    f_a = h5py.File(f'{cfg.val_dir}{cfg.save_part}/pattern_corr_ts_{str(cfg.resume_iter)}_assimilation_full.hdf5', 'r')
    
    corr_ts_mean_o = np.squeeze(np.array(f_o.get('corr_ts_mean')))
    corr_ts_f_o = np.squeeze(np.array(f_o.get('corr_ts_f')))
    corr_ts_c_o = np.squeeze(np.array(f_o.get('corr_ts_c')))
   
    corr_ts_mean_a = np.squeeze(np.array(f_a.get('corr_ts_mean')))
    corr_ts_f_a = np.squeeze(np.array(f_a.get('corr_ts_f')))
    corr_ts_c_a =  np.squeeze(np.array(f_a.get('corr_ts_c')))

    ticks = np.arange(0, len(corr_ts_mean_a), 12*5)
    labels = np.arange(1958, 2021, 5) 

    plt.figure(figsize=(10, 6))
    plt.plot(corr_ts_mean_o, 'x', label='Pattern Correlation mean')
    #plt.plot(corr_ts_f_o, label='Pattern Correlation F')
    #plt.plot(corr_ts_c_o, label='Pattern Correlation C')
    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title('Pattern Correlation of Ocean Heat Content Observations Reconstruction')
    plt.xlabel('Time in years')
    plt.ylabel(f'Pattern Correlation as ACC')
    plt.savefig(f'../Asi_maskiert/pdfs/validation/{part}/validation_std_timeseries_{str(cfg.resume_iter)}_observations.pdf')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(corr_ts_mean_a, 'x', label='Pattern Correlation mean')
    plt.plot(corr_ts_f_a, 'x', label='Pattern Correlation F')
    plt.plot(corr_ts_c_a, 'x', label='Pattern Correlation C')
    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title('Pattern Correlation of Ocean Heat Content Assimilation Reconstruction')
    plt.xlabel('Time in years')
    plt.ylabel(f'Pattern Correlation as ACC')
    plt.savefig(f'../Asi_maskiert/pdfs/validation/{part}/validation_std_timeseries_{str(cfg.resume_iter)}_assimilation.pdf')
    plt.show()


     


cfg.set_train_args()
#masked_output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=0)
#output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=0, mode='Observations')
#output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=0, mode='Assimilation')
#output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=1, mode='Observations')
#output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=1, mode='Assimilation')
#output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=9, mode='Observations')
#output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=9, mode='Assimilation')
#output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=19, mode='Observations')
#output_vis(cfg.save_part, str(cfg.resume_iter), time=cfg.val_interval, depth=19, mode='Assimilation')
#hc_plotting(cfg.save_part, cfg.resume_iter, time=cfg.val_interval, argo='full', mean='monthly')
#correlation_plotting(cfg.save_part, str(cfg.resume_iter), depth=0)
timeseries_plotting(cfg.save_part, cfg.resume_iter, argo='full', mean='annual')
pattern_corr_plot(cfg.save_part)
#std_plotting('part_1', 200000, 'full', del_t=2*12)



#vis_single(753, '../Asi_maskiert/original_image/', 'Image_3d_newgrid', 'r1011_shuffle_newgrid/short_val/Maske_1970_1985r1011_shuffle_newgrid/short_val/Maske_1970_1985Argo-era', 'image', 'image', 'North Atlantic Assimilation October 2020')
#vis_single(9, '../Asi_maskiert/original_masks/', 'Maske_2020_newgrid', 'pre-Argo-era', 'mask', 'mask', 'North Atlantic Observations October 2020')

#vis_single(1, '../Asi_maskiert/results/images/r1011_shuffle_newgrid/short_val/Maske_1970_1985/', 'test_700000', 'pre-argo-era', 'output', 'mask', 'Pre-Argo-Era Masks')