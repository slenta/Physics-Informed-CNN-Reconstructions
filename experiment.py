#creating timeseries from all assimilation ensemble members and ensemble mean
import numpy as np
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

depth=20
part='part_16'
iteration=1000000

fa = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iteration}_assimilation_full.hdf5', 'r')
fo = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iteration}_observations_full.hdf5', 'r')
fm = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
    
continent_mask = np.array(fm.get('continent_mask'))

mask = np.array(fa.get('mask')[:, 0, :, :]) * continent_mask
image_o = np.array(fo.get('image')[:, 0,:, :]) * continent_mask

mask_grey = np.where(mask==0, np.NaN, mask) * continent_mask
save_dir = f'../Asi_maskiert/pdfs/validation/{part}/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
fig.suptitle('Anomaly North Atlantic Heat Content')
plt.subplot(1, 2, 1)
plt.title(f'Preargo Observations: January 1958')
current_cmap = plt.cm.coolwarm
current_cmap.set_bad(color='gray')
plt.imshow(image_o[0] * mask_grey[0], cmap=current_cmap, vmin=-3, vmax=3, aspect='auto', interpolation=None)
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
#plt.colorbar(label='Temperature in °C')
plt.subplot(1, 2, 2)
plt.title(f'Argo Observations: August 2020')
im2 = plt.imshow(image_o[751] * mask_grey[751], cmap = 'coolwarm', vmin=-3, vmax=3, aspect = 'auto', interpolation=None)
plt.colorbar()
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
fig.savefig(f'../Asi_maskiert/pdfs/validation/{part}/Masks_{iteration}.pdf', dpi = fig.dpi)
plt.show()


gt_mean, std_gt, hc_all = evalu.hc_ensemble_mean_std(cfg.im_dir, name='Image_r', members=16)
length = hc_all.shape[1]
ticks = np.arange(0, length, 12*5)
labels = np.arange(1958, 2021, 5) 
plt.figure(figsize=(10, 6))
for i in range(16):
    globals()[f'r{i}'] = hc_all[i, :]
    globals()[f'r{i}'] = evalu.running_mean_std(globals()[f'r{i}'], mode='mean', del_t = 12)    
    plt.plot(globals()[f'r{i}'])
plt.xticks(ticks=ticks, labels=labels)
plt.xlim(0, len(gt_mean))
plt.title('Assimilation Ensemble Spread: 1958 -- 2020')
plt.grid()
plt.ylabel('Heat Content [J/m²]')
plt.xlabel('Time [years]')
plt.savefig(f'../Asi_maskiert/pdfs/validation/{part}/Ensemble_spread.pdf', dpi = fig.dpi)
plt.show()

f_a = h5py.File(f'../Asi_maskiert/results/validation/{part}/validation_{iteration}_assimilation_full.hdf5', 'r')
f_ah = h5py.File(f'../Asi_maskiert/results/validation/{part}/heatcontent_{iteration}_assimilation_full.hdf5', 'r')
    
fm = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
continent_mask = np.array(fm.get('continent_mask'))

hc_a = np.array(f_ah.get('net_ts')) * continent_mask
hc_gt = np.array(f_ah.get('gt_ts')) * continent_mask

correlation_argo_a, sig_argo_a = evalu.correlation(hc_a[552:], hc_gt[552:])
correlation_preargo_a, sig_preargo_a = evalu.correlation(hc_a[:552], hc_gt[:552])

acc_mean_argo_a = pearsonr(np.nanmean(np.nanmean(hc_gt[552:], axis=1), axis=1), np.nanmean(np.nanmean(hc_a[552:], axis=1), axis=1))[0]
acc_mean_preargo_a = pearsonr(np.nanmean(np.nanmean(hc_gt[:552], axis=1), axis=1), np.nanmean(np.nanmean(hc_a[:552], axis=1), axis=1))[0]

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
fig.suptitle('Anomaly North Atlantic Heat Content')
plt.subplot(1, 2, 1)
plt.title(f'Argo Reconstruction Correlation: {acc_mean_argo_a:.2f}')
current_cmap = plt.cm.coolwarm
current_cmap.set_bad(color='gray')
plt.scatter(sig_argo_a[1], sig_argo_a[0], c='black', s=0.7, marker='.', alpha=0.5)
im3 = plt.imshow(correlation_argo_a, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
plt.subplot(1, 2, 2)
plt.title(f'Preargo Reconstruction Correlation: {acc_mean_preargo_a:.2f}')
current_cmap = plt.cm.coolwarm
current_cmap.set_bad(color='gray')
plt.scatter(sig_preargo_a[1], sig_preargo_a[0], c='black', s=0.7, marker='.', alpha=0.5)
im3 = plt.imshow(correlation_preargo_a, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.xlabel('Transformed Longitudes')
plt.ylabel('Transformed Latitudes')
plt.colorbar(label='Annomaly Correlation')
fig.savefig(f'../Asi_maskiert/pdfs/validation/{part}/correlation_preargo_argo.pdf', dpi = fig.dpi)
plt.show()



####################pdfs argo/preargo    
del_t = 1
argo=cfg.mask_argo
n_windows = 1

f = h5py.File(f'{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_assimilation_{argo}.hdf5', 'r')
fo = h5py.File(f'{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_observations_{argo}.hdf5', 'r')

hc_assi = np.array(f.get('net_ts'))
hc_gt = np.array(f.get('gt_ts'))
hc_obs = np.array(fo.get('net_ts'))

#calculate running mean, if necessary
hc_assi = evalu.running_mean_std(hc_assi, mode='mean', del_t=del_t) 
hc_gt = evalu.running_mean_std(hc_gt, mode='mean', del_t=del_t) 
hc_obs = evalu.running_mean_std(hc_obs, mode='mean', del_t=del_t)
len_w = len(hc_assi)//n_windows
for i in range(n_windows):
    globals()[f'hc_a_{str(i)}'] = hc_assi[len_w*i:len_w * (i + 1)]
    globals()[f'hc_o_{str(i)}'] = hc_obs[len_w*i:len_w * (i + 1)]
    globals()[f'hc_gt_{str(i)}'] = hc_gt[len_w*i:len_w * (i + 1)]

    globals()[f'error_a_{str(i)}'] = np.sqrt((globals()[f'hc_a_{str(i)}'] - globals()[f'hc_gt_{str(i)}'])**2)       
    globals()[f'error_o_{str(i)}'] = np.sqrt((globals()[f'hc_o_{str(i)}'] - globals()[f'hc_gt_{str(i)}'])**2)       


    globals()[f'pdf_a_{str(i)}'] = norm.pdf(np.sort(globals()[f'error_a_{str(i)}']), np.mean(globals()[f'error_a_{str(i)}']), np.std(globals()[f'error_a_{str(i)}']))
    globals()[f'pdf_o_{str(i)}'] = norm.pdf(np.sort(globals()[f'error_o_{str(i)}']), np.mean(globals()[f'error_o_{str(i)}']), np.std(globals()[f'error_o_{str(i)}']))
    
plt.title('Error PDFs')
if n_windows!=1:
    for i in range(n_windows):
        plt.plot(np.sort(globals()[f'error_a_{str(i)}']), globals()[f'pdf_a_{str(i)}'], label=f'Assimilation Reconstruction Error Pdf {str(i)}')
        #plt.plot(np.sort(globals()[f'error_o_{str(i)}']), globals()[f'pdf_o_{str(i)}'], label=f'Observations Reconstruction Error Pdf {str(i)}')
plt.grid()
plt.ylabel('Probability Density')
plt.xlabel('Absolute Error of Reconstruction')
plt.legend()
plt.savefig(f'../Asi_maskiert/pdfs/validation/{cfg.save_part}/error_pdfs.pdf')
plt.show()
#for member in range(1, 17):
#    print(member)
#
#    ifile = f'/work/uo1075/decadal_system_mpi-esm-lr_enkf/data/MPI-ESM1-2-LR/asSEIKERAf/Omon/thetao/r{member}i8p4/thetao_Omon_MPI-ESM-LR_asSEIKERAf_r{member}i8p4_195801-202010.nc'
#    ofile = f'../Asi_maskiert/original_image/Image_r{member}_newgrid.nc'
#
#    cdo.sellonlatbox(-65, -5, 20, 69, input = ifile, output = ofile)
#
#    ds = xr.load_dataset(ofile, decode_times=False)
#
#    f = h5py.File('../Asi_maskiert/original_image/baseline_climatologyfull.hdf5', 'r')
#    tos_mean = f.get('sst_mean')
#            
#    tos = ds.thetao.values
#            
#    for i in range(len(tos)):
#        tos[i] = tos[i] - tos_mean[i%12]
#
#    tos = np.nan_to_num(tos, nan=0)
#
#    n = tos.shape
#    rest = np.zeros((n[0], n[1], cfg.image_size - n[2], n[3]))
#    tos = np.concatenate((tos, rest), axis=2)
#    n = tos.shape
#    rest2 = np.zeros((n[0], n[1], n[2], cfg.image_size - n[3]))
#    tos = np.concatenate((tos, rest2), axis=3)
#
#    tos = tos[:, :depth, :, :]
#
#    #val_cut
#    ds_compare = xr.load_dataset(f'{cfg.im_dir}Image_r9_newgrid.nc')
#
#    lat = np.array(ds_compare.lat.values)
#    lon = np.array(ds_compare.lon.values)
#    time = np.array(ds_compare.time.values)
#
#    n = tos.shape
#    lon_out = np.arange(cfg.lon1, cfg.lon2)
#    lat_out = np.arange(cfg.lat1, cfg.lat2)
#    tos_new = np.zeros(shape=(n[0], n[1], len(lat_out), len(lon_out)), dtype='float32')
#
#    for la in lat_out:
#        for lo in lon_out: 
#            x_lon, y_lon = np.where(np.round(lon)==lo)
#            x_lat, y_lat = np.where(np.round(lat)==la)
#            x_out = []
#            y_out = []
#            for x, y in zip(x_lon, y_lon):
#                for a, b in zip(x_lat, y_lat):
#                    if (x, y) == (a, b):
#                        x_out.append(x)
#                        y_out.append(y)
#            for i in range(len(time)):
#                for j in range(depth):
#                    tos_new[i, j, la - min(lat_out), lo - min(lon_out)] = np.mean([tos[i, j, x, y] for x, y in zip(x_out, y_out)])
#    
#    tos_new = tos_new[:, :, ::-1, :]
#
#    prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
#    depth_steps = prepo.depths()
#
#    continent_mask_cut = np.where(tos_new[0, 0, :, :]==0, np.nan, 1)
#    continent_mask = np.where(tos[0, 0, :, :]==0, np.nan, 1)
#
#    tos_new = np.nanmean(tos_new * continent_mask_cut, axis=(3, 2))
#    tos = np.nanmean(tos * continent_mask, axis=(3, 2))
#
#    hc_cut = np.zeros(n[0])
#    hc = np.zeros(n[0])
#    rho = 1025  #density of seawater
#    shc = 3850  #specific heat capacity of seawater
#
#    for j in range(n[0]):
#        hc[j] = np.sum([(depth_steps[k] - depth_steps[k-1])*tos[j, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * tos[j, 0] * rho * shc
#        hc_cut[j] = np.sum([(depth_steps[k] - depth_steps[k-1])*tos_new[j, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * tos_new[j, 0] * rho * shc
#
#
#    #save both full and val_cut
#    h5_name = f'../Asi_maskiert/original_image/Image_r{member}_anomalies_depth_full_20.hdf5'
#    f = h5py.File(h5_name, 'w')
#    f.create_dataset('hc', shape=hc.shape, data=hc)
#    f.create_dataset('hc_cut', shape=hc_cut.shape, data=hc_cut)
#    f.close()

